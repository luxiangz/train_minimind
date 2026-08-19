from transformers import PretrainedConfig
import math
import torch

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        # TODO
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

class RMSNorm(torch.nn.Module):
    # x/sqrt(variance + epsilon)*gamma 残差结构本身已经"平移不变"，均值偏移会被残差吸收——所以减均值这步是冗余的
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps
        # 融合算子优雅降级：torch>=2.4 提供 F.rms_norm（单 kernel，约 4.7x 加速）
        # 检测一次即可，避免每次 forward 重复开销
        self._use_fused = hasattr(torch.nn.functional, 'rms_norm') #TODO 记住

    def norm(self, x):
        #rsqrt  cuda上的
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) 

    def forward(self, x):
        if self._use_fused:
            # 融合分支：单 kernel 一次读完 x、算完写回，零中间张量
            return torch.nn.functional.rms_norm(
                x, (self.weight.shape[0],), self.weight, self.variance_epsilon
            )
        # 手写回退分支：fp32 计算保证精度，再转回输入精度
        return (self.weight * self.norm(x.float())).type_as(x)
# 与计算所有位置的cos/sin，然后查表
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None): #cis(x) = cos(x) + i·sin(x)，复数的极坐标写法
    # θᵢ = base^(-2i/dim)   
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim)), 1.0
    # TODO YaRN PASS
    t = torch.arange(end, device=freqs.device)
    # 外积矩阵 每个位置×每个频率 angle[m, i] = m × θᵢ 
    freqs = torch.outer(t, freqs).float() # (end, dim//2) 外积
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # q_m · cos(angle) + rotate_half(q_m) · sin(angle) 
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1) #TODO
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

class Attention(torch.nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.wq = torch.nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.wk = torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wv = torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wo = torch.nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = torch.nn.Dropout(config.dropout)
        self.resid_dropout = torch.nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 将kv重复到多个头
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if (self.flash                                # 环境支持 FA
            and (seq_len > 1)                         # 序列长度 > 1
            and (not self.is_causal or past_key_value is None)  # 因果 or 无缓存
            and (attention_mask is None or torch.all(attention_mask == 1))):  # 无自定义掩码
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv