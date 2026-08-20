import os
import sys
import time
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import datasets
import argparse
import torch
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    use_cuda_events = (device_type == "cuda")
    if use_cuda_events:
        ev_step_start = torch.cuda.Event(enable_timing=True)
        ev_load_end   = torch.cuda.Event(enable_timing=True)
        ev_fwd_start  = torch.cuda.Event(enable_timing=True)
        ev_fwd_end    = torch.cuda.Event(enable_timing=True)
        ev_bwd_start  = torch.cuda.Event(enable_timing=True)
        ev_bwd_end    = torch.cuda.Event(enable_timing=True)
        ev_opt_start  = torch.cuda.Event(enable_timing=True)
        ev_step_end   = torch.cuda.Event(enable_timing=True)

    for step, (input_ids, labels) in enumerate(loader, start=start_step+1):
        if use_cuda_events:
            ev_step_start.record()              # 端到端开始（GPU 事件）
        else:
            step_start_time = time.time()
        load_start = time.time()                # 数据加载 + 搬运计时（CPU 侧）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        load_time = time.time() - load_start    # ① 数据加载耗时（含 .to 搬运）
        last_step =step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if use_cuda_events:
            ev_load_end.record()            # 数据加载结束（GPU 事件）
            ev_fwd_start.record()           # 前向开始
        with autocast_ctx:
            res = model(input_ids,labels = labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        if use_cuda_events:
            ev_fwd_end.record()             # 前向结束
            ev_bwd_start.record()           # 反向开始
        scaler.scale(loss).backward()       # 反向（DDP 下含梯度同步 all-reduce）
        if use_cuda_events:
            ev_bwd_end.record()             # 反向（含通信）结束
        else:
            fwd_time = 0.0; bwd_time = 0.0

        if use_cuda_events:
            ev_opt_start.record()           # 优化器开始
        opt_start = time.time()
        if step % args.accumulation_steps == 0:
            # 还原梯度 + 检查 inf/nan
            scaler.unscale_(optimizer)
            # 按照全局范数裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 检查后更新梯度
            scaler.step(optimizer)
            # 根据前面的结果 动态调整缩放系数 k
            scaler.update()
            # 清空梯度
            optimizer.zero_grad(set_to_none=True)
        opt_time = time.time() - opt_start   # ⑤ 优化器耗时
        if use_cuda_events:
            ev_opt_end = torch.cuda.Event(enable_timing=True)
            ev_opt_end.record()
            ev_step_end.record()
            torch.cuda.synchronize()         # 同步，确保事件计时完成

            # 用 GPU 事件精确计算各段耗时（毫秒）
            step_time = ev_step_start.elapsed_time(ev_step_end) / 1000.0       # 端到端（秒）
            load_time_gpu = ev_step_start.elapsed_time(ev_load_end) / 1000.0   # 数据加载+搬运
            fwd_time = ev_fwd_start.elapsed_time(ev_fwd_end) / 1000.0          # 前向
            bwd_time = ev_bwd_start.elapsed_time(ev_bwd_end) / 1000.0          # 反向（含通信）
            opt_time_gpu = ev_opt_start.elapsed_time(ev_opt_end) / 1000.0      # 优化器
            # 反向里的"纯反向+梯度同步"粗略拆：反向段内除前向外的部分
            bwd_comm_time = max(bwd_time - fwd_time * 0.0, 0.0)                # 反向含通信（DDP）
            step_time = max(step_time, 1e-9)
            model_time = fwd_time + bwd_time
        else:
            step_time = time.time() - step_start_time
            model_time = fwd_time + bwd_time
            bwd_comm_time = 0.0
        batch_count = len(input_ids)
        model_ratio = model_time / max(step_time, 1e-9)
        modelfwdratio = fwd_time / max(step_time, 1e-9)
        modelbwdratio = bwd_time / max(step_time, 1e-9)

        # 每个 step 都用 wandb 记录性能指标
        if step % 2 == 0 or step == iters:
            if wandb:
                wandb.log({
                    "batch_count": batch_count,          # 每 step batch 数（样本数）
                    "step_time_s": step_time,            # 端到端 step 耗时（秒）
                    "model_time_s": model_time,          # 模型总耗时（前向+反向，秒）
                    "fwd_time_s": fwd_time,              # 前向耗时（秒）
                    "bwd_time_s": bwd_time,              # 反向耗时（秒，DDP 含梯度同步）
                    "comm_time_s": bwd_comm_time,        # 梯度同步/通信（多卡 DDP 时有效）
                    "load_time_s": load_time,            # 数据加载 + 搬运耗时（秒）
                    "opt_time_s": opt_time,              # 优化器耗时（秒）
                    "fwd_ratio": modelfwdratio,          # 前向占模型比例
                    "bwd_ratio": modelbwdratio,          # 反向占模型比例
                    "model_time_ratio": model_ratio,     # 模型计算占端到端比例
                })

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            Logger(f'  [Perf] batch={batch_count}, step={step_time*1000:.1f}ms | load={load_time*1000:.1f} fwd={fwd_time*1000:.1f} bwd={bwd_time*1000:.1f} comm={bwd_comm_time*1000:.1f} opt={opt_time*1000:.1f} | model_ratio={model_ratio*100:.1f}%')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})
        # 保存模型
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 初始化环境和随机种子
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # 配置目录、模型参数、检查ckp
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # 设置混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # 配置wandb
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # 定义模型、数据、优化器
    model, tokenizer = init_model(lm_config, args.from_weight)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 多卡时，每张卡由一个进程驱动，数据必须分成 N 份、每卡一份、互不重叠
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler（梯度缩放器）fp16 训练时用"放大梯度再更新"来防止梯度下溢，bf16 时不需要（禁用）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 从ckp回复状态
    # 数据集定义： 题库
    # batch_size: 一套卷子中题目数量
    # epoch 定义：做完题库的所有卷子
    # step/batch 定义：做完一套卷子
    # 优化器更新accumulation_steps：做了几套卷子再更新

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # 编译和分布式包装
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch) # 多卡时让 DistributedSampler 每个 epoch 重新洗牌（否则所有 epoch 数据顺序相同），单卡时什么都不做。
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist() # 根据数据集大小生成随机乱序的index的list
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True) # loader 就是"已经分好 batch 的数据集合，是一个迭代器、
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()








