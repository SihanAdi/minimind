import math
from typing import Optional
import torch
from torch import device, nn
from torch.nn import functional as F
from torch.nn import init

# 将激活函数名称映射到具体实现函数
from transformers.activations import ACT2FN
# GenerationMixin 提供文本生成功能; 实现 generate() 方法及相关生成策略; 支持 beam search、sampling、contrastive search 等生成方式
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu', # 相比ReLU有更好的梯度特性，在Transformer中表现优秀，LLaMA、GPT等现代模型的首选激活函数
            hidden_size: int = 512, # 足够表达复杂模式，同时计算效率较高，适合中等规模的模型
            intermediate_size: int = None,
            max_position_embeddings: int = 32768, # 32K上下文
            
            # 8:2比例：使用分组查询注意力(GQA)，减少KV缓存内存占用
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            
            num_hidden_layers: int = 8,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # # MOE, When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            # 4+1配置：提供专家多样性，同时有共享专家处理通用知识
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # MOE, When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class RMSNorm(nn.Module):
    """
    RMSNorm 解决 LayerNorm 被存在的计算上的冗余
        - LayerNorm 关键部分在于 重新中心化 和 重新缩放 操作，而其中的重新中心化（减去均值）可能不是必须的
    RMSNorm 只缩放，不中心化：使用均方根 来代替标准差进行缩放
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def apply_yarn(dim, orig_max, factor, beta_fast, beta_slow, freqs):
    """
    YaRN: 高效扩展基于 RoPE 的模型上下文窗口，RoPE中不同频率的维度对位置信息的编码方式不同，不能"一刀切"地处理
    高频维度：具备周期性和鲁棒的，可以直接按原始方式计算 - 波长 ≤ 原始最大长度 小β
    低频维度：承载了全局绝对位置信息，需要进行内插，压缩到模型熟悉的范围内 - 波长 > 原始最大长度 大β

    YaRN的标准缩放公式：scale_low_freq = (beta * factor - beta + 1) / (beta * factor)
        - 解决线形插值把所有位置信息压缩到原来的 1/factor，导致所有维度都被同等对待，高频维度损失严重

    相关信息：
        - RoPE的旋转角度符合物理学定义上的角频率：角度随时间（或位置）的变化率
        - 波长 = 波速 × 周期 = 周期 / 角频率；波长就是从起点到完成一次完整旋转所经过的位置距离
            - 如果波长 > 原始最大长度，说明在模型的训练范围内，这个维度从未完成过完整周期
    """
    # 找到第一个高频维度的索引位置
    first_high_freq_idx = next(
        (i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2
    )

    # 为每个维度分配一个权重，用于在 beta_slow 和 beta_fast 之间进行平滑插值
    weights = torch.arange(0, dim // 2, device=freqs.device).float() / max(
        dim // 2 - 1, 1
    )

    # 计算每个维度的beta值: 低频维度（weights接近0）：beta ≈ beta_slow, 高频维度（weights接近1）：beta ≈ beta_fast, 中间维度：平滑过渡
    beta = beta_slow + (beta_fast - beta_slow) * weights

    # scale_low_freq = (beta * factor - beta + 1) / (beta * factor)
    scale = torch.where(
        torch.arange(0, dim // 2, device=freqs.device) < first_high_freq_idx,
        (beta * factor - beta + 1) / (beta * factor),
        1.0 / factor,
    )

    freqs = freqs * scale
    return freqs


def precompute_freqs_cis(dim, end=int(32 * 1024), rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    if rope_scaling:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0),
            rope_scaling.get("beta_slow", 1.0),
        )
        if end / orig_max > 1.0:
            freqs = apply_yarn(dim, orig_max, factor, beta_fast, beta_slow, freqs)
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs).float()

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)
    # 输出size[sequence len, head_dim]
    return freqs_cos, freqs_sin


def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    """
    通过旋转矩阵对词向量在高维空间中进行旋转，旋转的角度取决于该 token 的绝对位置
        - 旋转: 天然地包含了绝对位置信息，并且神奇地保持了相对位置的规律性
    两个词，一个在位置 m，一个在位置 n: 位置 m 的词向量被旋转了 m*θ; 位置 n 的词向量被旋转了 n*θ
        - 旋转后的两个向量的点积，只与它们原始向量的点积和相对位置 (m-n) 有关
        - 只依赖于词本身的语义和它们的相对距离，而与它们的绝对位置无关
    词向量的维度 d 通常很高, RoPE：将高维向量视为 d/2 个二维向量的拼接
        - [x1, x2, x3, x4]。每一对 (x1, x3) 构成一个二维子空间，(x2, x4) 构成另一个...
        - 对于第 i 个二维子空间，我们使用一个不同的基础旋转角度 θ_i。通常，θ_i 会随着 i 的增大而减小
    只对 Query 和 Key 向量应用位置编码，而 Value 向量保持不变
    """
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    # 输出size[b, s, n_heads, head_dim]
    return q_embed, k_embed


def rotate_half(x):
    """
    [A, B] -> [-B, A]
    """
    half_index = x.shape[-1] // 2
    return torch.cat((-x[..., half_index:], x[..., :half_index]), dim=-1)


def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    batch_size, seq_len, n_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_heads * n_rep, head_dim)
        )


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 注意力头相关参数
        # GQA 分组查询注意力：多个query头共享相同的key和value头
        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )
        assert args.num_attention_heads % args.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        self.n_local_heads = args.num_attention_heads  # Query的总头数
        self.n_rep = (
            self.n_local_heads // self.num_key_value_heads
        )  # 每个KV头对应的Q头数
        # 维度参数
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 投影层
        self.q_proj = nn.Linear(
            args.hidden_size, self.head_dim * self.n_local_heads, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.head_dim * self.num_key_value_heads, bias=False
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.n_local_heads, args.hidden_size, bias=False
        )
        # dropout
        self.dropout = args.dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 优化参数
        self.flash = args.flash_attn and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embedding,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        batch_size, seq_len, _ = x.shape
        # 计算qkv
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # 对qk计算rope
        cos, sin = position_embedding  # [seq_len, head_dim]
        q, k = apply_rope(q, k, cos[:seq_len], sin[:seq_len])

        # kv cache实现
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        if use_cache:
            past_key_value = (k, v)
        else:
            past_key_value = None

        # 每个头都能独立处理自己的 [seq_len, head_dim] 矩阵, 提高并行效率
        q = q.transpose(1, 2)  # batch_size, n_local_heads, seq_len, head_dim
        k, v = (
            repeat_kv(k, self.n_rep).transpose(1, 2),
            repeat_kv(v, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):  # 不支持批量序列对齐padding
            attention_mask = (
                None
                if attention_mask is None
                else attention_mask.view(batch_size, 1, 1, -1)
                .expand(batch_size, self.n_local_heads, seq_len, -1)
                .bool()
            )

            output = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (
                q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            )  # [batch_size, n_local_heads, seq_len, seq_len]

            # causal mask + scores
            scores += (
                torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            # attention_mask 处理由于输入数据序列长度不一致所产生的padding部分
            if attention_mask is not None:
                extend_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores += (1.0 - extend_attention_mask) * -1e9  # 1: not mask; 0: mask

            scores = F.softmax(scores, dim=-1).type_as(q)
            scores = self.attn_dropout(scores)

            output = scores @ v  # [batch_size, n_local_heads, seq_len, head_dim]

        output = output.transpose(2, 1).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_key_value


class FeedForward(nn.Module):
    """
    门控前馈网络
    相较传统前馈网络，门控前馈网络在激活函数前增加了一个门控机制，使得模型在训练过程中可以更有效地控制信息的流动，从而提高模型的训练效率和泛化能力。
    """

    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            """
            传统FFN参数量：hidden × intermediate × 2
                - x → Linear(hidden→intermediate) → Activation → Linear(intermediate→hidden) → Output
            门控FFN参数量：hidden × intermediate × 3
            通常intermediate_old = 4 × hidden
            """
            intermediate_size = int(config.hidden_size * 4 * (2 / 3))
            """
            hidden_size = 512
            理论值 = 512 × 8/3 ≈ 1365.33
            对齐后 = 64 × ceil(1365.33/64) = 64 × 22 = 1408
            """
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        x → Gate_Linear → Activation ↘
                                    Multiply → Down_Linear → Output
        x → Up_Linear   →───────────↗
        """
        return self.dropout(
            self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))
        )


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_id = layer_id

        self.self_attn = Attention(config)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        position_embedding,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, past_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embedding,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, past_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # 使用 register_buffer 而不是直接简单的实例变量原因：自动转移到设备
        # persistent=False 这个 Buffer 不会被保存到模型状态字典中，因为freqs_cos 和 freqs_sin 是完全可以通过配置参数重新计算，显著减小模型文件大小
        # 自动注册为当前对象的属性
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape

        # 适配多种输入格式的 past_key_values
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or ([None] * len(self.layers))

        # 确定当前处理的起始位置 - 增量训练
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # 准备输入数据
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embedding = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []
        for _, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embedding,
                past_key_value,
                use_cache,
                attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        # MoE机制
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )


        return hidden_states, presents, aux_loss


class MoEGate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.topk = config.num_experts_per_topk  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 总专家数

        self.scoring_func = config.scoring_func
        self.alpha = (
            config.aux_loss_alpha
        )  # 确保专家负载均衡，避免某些专家过载而其他专家闲置
        self.seq_aux = (
            config.seq_aux
        )  # 如果为True，在序列级别计算负载均衡损失；否则在token级别计算

        self.norm_topk_rob = config.norm_topk_prob  # 是否对top-k概率进行归一化
        self.gating_dim = config.hidden_size  # 门控网络的输入维度
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )  # [n_routed_experts, hidden_size]
        self.reset_parameters()

    def reset_parameters(self):
        """
        适用于：
            - 线性层（全连接层）
            - 卷积层
            - 使用Leaky ReLU或类似激活函数的网络
        """
        return init.kaiming_uniform_(
            self.weight, a=math.sqrt(5)
        )  # 源于原始论文中的推荐，它对于某些网络结构能够提供更好的训练稳定性

    def forward(self, hidden_states):
        """目标负载均衡"""
        # 处理输入
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(
            -1, hidden_dim
        )  # [batch_size * seq_len, hidden_dim]

        # 计算分数
        logits = F.linear(
            input=hidden_states, weight=self.weight, bias=None
        )  # [batch_size * seq_len, n_routed_experts]
        if self.scoring_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(
                f"Insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # 计算topk个专家的权重和idx，并归一化权重
        topk_weight, topk_idx = torch.topk(scores, k=self.topk, dim=-1, sorted=False)

        if self.topk > 1 and self.norm_topk_rob:
            denominator = torch.sum(topk_weight, dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            topk_idx_for_aux_loss = topk_idx.view(
                batch_size, -1
            )  # [batch_size, seq_len * top_k]
            # 负载均衡辅助损失，目的是确保所有专家都能被均衡使用
            if self.seq_aux:
                # 序列级别辅助损失
                score_for_seq_aux = scores.view(batch_size, seq_len, -1)

                # 计算专家使用频率
                ce = torch.zeros(
                    batch_size, self.n_routed_experts, device=hidden_states.device
                )

                """
                在专家索引位置累加1，统计每个专家被选中的次数，并归一化
                
                例如：
                batch_size = 2; seq_len = 3; topk = 2; n_routed_experts = 4
                
                ce: [batch_size, n_routed_experts]
                [
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]
                ]
                
                topk_idx_for_aux_loss: [batch_size, seq_len * top_k]
                [
                    [1, 0, 2, 3, 1, 0],
                    [3, 1, 1, 2, 0, 3]
                ]
                
                src: [batch_size, seq_len * topk]
                [
                    [1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1.]
                ]
                
                dim = 1具体计算过程：
                    - 对于batch 0：
                        index[0,0]=1 → ce[0,1] += 1 → ce[0,1] = 1
                        index[0,1]=0 → ce[0,0] += 1 → ce[0,0] = 1
                        index[0,2]=2 → ce[0,2] += 1 → ce[0,2] = 1
                        index[0,3]=3 → ce[0,3] += 1 → ce[0,3] = 1
                        index[0,4]=1 → ce[0,1] += 1 → ce[0,1] = 2
                        index[0,5]=0 → ce[0,0] += 1 → ce[0,0] = 2
                    - 对于batch 1：
                        index[1,0]=3 → ce[1,3] += 1 → ce[1,3] = 1
                        index[1,1]=1 → ce[1,1] += 1 → ce[1,1] = 1
                        index[1,2]=1 → ce[1,1] += 1 → ce[1,1] = 2
                        index[1,3]=2 → ce[1,2] += 1 → ce[1,2] = 1
                        index[1,4]=0 → ce[1,0] += 1 → ce[1,0] = 1
                        index[1,5]=3 → ce[1,3] += 1 → ce[1,3] = 2
                输出ce:
                [
                    [2., 2., 1., 1.],
                    [1., 2., 1., 2.]
                ]
                """
                ce.scatter_add_(
                    dim=1,  # 沿专家维度操作
                    index=topk_idx_for_aux_loss,
                    src=torch.ones(
                        batch_size, seq_len * self.topk, device=hidden_states.device
                    ),
                ).div_(
                    # 原始计数：每个专家被选中的次数 = count_i
                    # 总的选择次数：total_choices = bsz * seq_len * aux_topk
                    # 理想情况下每个专家应该被选中的次数：ideal_per_expert = total_choices / n_routed_experts
                    seq_len * self.topk / self.n_routed_experts
                )

                # 计算损失
                expert_scores = score_for_seq_aux.mean(
                    dim=1
                )  # [batch_size, n_routed_experts]
                aux_loss = (ce * expert_scores).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )  # [batch_size * seq_len * top_k, n_routed_experts]
                ce = mask_ce.float().mean(0)  # [n_routed_experts] 每个专家被选中的概率
                Pi = scores.mean(
                    0
                )  # [batch_size * seq_len, n_routed_experts] -> [n_routed_experts] 计算专家得分期望
                """
                fi = 1.0：使用频率正好等于理想值
                fi > 1.0：过度使用
                fi < 1.0：使用不足
                """
                """
                ce = 每个专家被选中的概率 = P_actual(i)
                理想情况下每个专家被选中的概率：P_ideal(i) = 1 / n_routed_experts
                归一化：fi = P_actual(i) / P_ideal(i) = P_actual(i) * n_routed_experts
                """
                fi = ce * self.n_routed_experts  # 归一化使用频率
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # 路由专家
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 共享专家
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )
        else:
            self.shared_experts = None
        # 专家router
        self.gate = MoEGate(config)
        
    def forward(self, x):
        identity = x
        batch_size, seq_len, hidden_dim = x.shape
        
        # 使用门控机制选择专家
        topk_idx, topk_weights, self.aux_lost = self.gate(x)    # [batch_size * seq_len, n_routed_experts]
        flat_topk_idx = topk_idx.view(-1)    # [batch_size * seq_len * n_routed_experts]
        x = x.view(-1, hidden_dim)    # [batch_size * seq_len, hidden_dim]
        
        if self.training:
            x = torch.repeat_interleave(x, repeats=self.config.n_routed_experts, dim=0)    # [n_routed_experts * batch_size * seq_len, hidden_dim]
            y = torch.empty_like(x, dtype=torch.float16)
            
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).type_as(y)
            
            # [batch_size * seq_len, n_routed_experts, hidden_dim] -> [batch_size * seq_len, hidden_dim]
            y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
            y = y.view((batch_size, seq_len, hidden_dim))
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weights.view(-1, 1)).view((batch_size, seq_len, hidden_dim))
        
        # 共享
        if self.shared_experts is not None:
            for expert in self.shared_experts:
                y += expert(identity)
                
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        x: [batch_size * seq_len, hidden_dim]
        flat_expert_indices: [batch_size * seq_len * n_routed_experts]
        flat_expert_weights: [batch_size * seq_len * n_routed_experts, 1] = flat_expert_weights^T
        
        argsort() 返回的是排序后(升序)的索引位置，而不是排序后的值本身
        """
        expert_cache = torch.zeros_like(x)
        
        # 按专家编号排列，所有相同专家的处理被集中在一起实现批量计算
        idxs = flat_expert_indices.argsort()
        
        # 统计每个专家处理的token数量，并计算累积和，得到每个专家的结束位置索引
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # 还原出原始的token索引
        token_idxs = idxs // self.config.num_experts_per_tok
        
        for i, end in enumerate(tokens_per_expert):
            """
            当tokens_per_expert = [6, 15, 20, 26]
            tokens_per_expert.shape[0]即为专家数量（此时为4）
            且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5, ...] 时
            意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token
            （每个token有可能被多个专家处理，这取决于num_experts_per_tok）
            接下来9个位置token_idxs[6:15] -> [4,  5, ...]属于专家1处理的token
            ...
            """
            # 确定专家处理的token范围
            start = 0 if i == 0 else tokens_per_expert[i - 1]
            if start == end:
                continue
            
            expert = self.experts[i]
            expert_token_idxs = token_idxs[start:end]
            expert_tokens = x[expert_token_idxs]
            
            output = expert(expert_tokens).type_as(expert_cache)
            output.mul_(output, flat_expert_weights[idxs[start:end]])
            
            # 将结果累加到对应的token位置
            expert_cache.scatter_add_(
                dim=0, 
                index=expert_token_idxs.view(-1, 1).repeat(1, x.shape[-1]), 
                src=output
            )
        return expert_cache


class MiniMindForCausalLm(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config=None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.llm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 将输入嵌入层（embed_tokens） 和输出层（lm_head） 的权重共享
        self.model.embed_tokens.weight = self.llm_head.weight
        
        """
        CausalLMOutputWithPast(数据容器类，专门用于因果语言模型的输出) 主要包含的字段：
        - loss: torch.FloatTensor | None  # 总损失值
        - logits: torch.FloatTensor       # 语言模型头的输出logits
        - past_key_values: tuple | None   # 用于生成的缓存键值对
        - hidden_states: tuple | None     # 所有隐藏状态（如果输出隐藏状态）
        - attentions: tuple | None        # 所有注意力权重（如果输出注意力）
        - aux_loss: torch.FloatTensor | None  # 辅助损失（如MoE模型）
        
        CausalLMOutputWithPast 的设计是为了与 GenerationMixin 的生成方法完美配合
        """
        self.OUT = CausalLMOutputWithPast()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        **args,
    ):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.llm_head(h[:,slice_indices,:])
        self.OUT.__setitem__('last_hidden_state', h) # 最后一个Transformer层的输出
        self.OUT.__setitem__('logits', logits) # 语言模型头的输出，用于预测下一个token
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
