# MoE Gate

## 基本概念回顾

### 负载均衡的目标
确保所有专家被均匀使用，避免：
- **专家过载**：某些专家处理过多token
- **专家闲置**：某些专家很少被使用

## 序列级别负载均衡损失

### 计算流程
```python
# 输入形状说明
# hidden_states: [batch_size, seq_len, hidden_size]
# topk_idx: [batch_size, seq_len, topk]

def sequence_level_aux_loss(hidden_states, topk_idx, scores):
    bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    
    # 1. 统计每个专家被选中的次数（按batch）
    ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
    topk_idx_flat = topk_idx.view(bsz, -1)  # [batch_size, seq_len * topk]
    
    ce.scatter_add_(1, topk_idx_flat, 
                   torch.ones(bsz, seq_len * self.topk, device=hidden_states.device))
    
    # 2. 归一化到理想负载
    ce.div_(seq_len * self.topk / self.n_routed_experts)  # [batch_size, num_experts]
    
    # 3. 计算专家得分均值
    expert_scores = scores.view(bsz, seq_len, -1).mean(dim=1)  # [batch_size, num_experts]
    
    # 4. 计算损失
    aux_loss = (ce * expert_scores).sum(dim=1).mean() * self.alpha
    
    return aux_loss
```

### 数学表达式
对于每个batch $b$：
$
\text{ce}_{b,j} = \frac{\text{count}_{b,j}}{\frac{\text{seq\_len} \times \text{topk}}{n\_experts}}
$

$
\text{aux\_loss} = \alpha \times \frac{1}{B} \sum_{b=1}^{B} \sum_{j=1}^{E} \text{ce}_{b,j} \times \text{score}_{b,j}
$

其中：
- $\text{count}_{b,j}$：batch $b$中专家$j$被选中的次数
- $\text{score}_{b,j}$：batch $b$中专家$j$的平均得分

### 特点分析
**优点：**
- 保持batch级别的独立性
- 对异常batch有一定的鲁棒性
- 计算相对高效

**缺点：**
- 对batch大小敏感
- 可能掩盖细粒度的负载不均衡

## Token级别负载均衡损失

### 计算流程
```python
def token_level_aux_loss(hidden_states, topk_idx, scores):
    bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    
    # 1. 创建one-hot编码（全局）
    topk_idx_flat = topk_idx.view(-1)  # [batch_size * seq_len * topk]
    mask_ce = F.one_hot(topk_idx_flat, num_classes=self.n_routed_experts)
    
    # 2. 计算全局使用频率
    ce = mask_ce.float().mean(0)  # [num_experts]
    
    # 3. 计算全局得分期望
    Pi = scores.mean(0)  # [num_experts]
    
    # 4. 归一化使用频率
    fi = ce * self.n_routed_experts  # [num_experts]
    
    # 5. 计算损失
    aux_loss = (Pi * fi).sum() * self.alpha
    
    return aux_loss
```

### 数学表达式
全局计算：
$
\text{ce}_j = \frac{1}{B \times L \times K} \sum_{b=1}^{B} \sum_{i=1}^{L} \sum_{k=1}^{K} \mathbb{1}[\text{topk\_idx}_{b,i,k} = j]
$

$
\text{Pi}_j = \frac{1}{B \times L \times K} \sum_{b=1}^{B} \sum_{i=1}^{L} \sum_{k=1}^{K} \text{score}_{b,i,j}
$

$
\text{fi}_j = \text{ce}_j \times E
$

$
\text{aux\_loss} = \alpha \times \sum_{j=1}^{E} \text{Pi}_j \times \text{fi}_j
$

### 特点分析
**优点：**
- 全局视角，更稳定的梯度
- 对batch大小不敏感
- 能捕捉细粒度的负载模式

**缺点：**
- 需要更多的内存（one-hot编码）
- 计算量稍大
- 可能被大batch主导

## 详细对比分析

### 1. 计算粒度对比
| 方面 | 序列级别 | Token级别 |
|------|----------|-----------|
| 统计单位 | 每个batch独立 | 所有token全局 |
| 损失计算 | 按batch平均 | 直接全局计算 |
| 视角 | 局部batch视角 | 全局数据分布视角 |

### 2. 适用场景

**选择序列级别的情况：**
- 小批量训练
- 内存受限环境
- 需要快速实验迭代

**选择Token级别的情况：**
- 大批量分布式训练
- 追求训练稳定性
- 有充足内存资源

## 实际效果示例

### 负载不均衡场景
假设有8个专家，但只有4个被频繁使用：

**序列级别可能产生：**
```
Batch1: [1.8, 1.5, 0.2, 0.1, 1.6, 1.9, 0.3, 0.2] → 损失较高
Batch2: [0.3, 0.2, 1.7, 1.8, 0.1, 0.2, 1.9, 1.8] → 损失较高
平均损失仍然较高，推动均衡化
```

**Token级别直接看到：**
```
全局: [1.2, 1.1, 0.9, 0.8, 1.3, 1.0, 0.7, 0.9] → 明显不均衡
持续产生高损失，强力推动均衡化
```

## 总结

两种方法都有效地解决了MoE负载均衡问题，但各有侧重：
- **序列级别**：计算高效，适合资源受限场景
- **Token级别**：训练稳定，适合大规模部署


# 为什么过载带来的损失增加**大于**闲置带来的损失减少。

## 数学原理分析

### 损失函数回顾
$
\text{aux\_loss} = \sum_{j=1}^{E} \text{Pi}_j \times \text{fi}_j = \sum_{j=1}^{E} \text{Pi}_j \times (\text{ce}_j \times E)
$

其中 $\text{fi}_j = \text{ce}_j \times E$

- 当 $\text{ce}_j = 1.0$（理想使用）时，$\text{fi}_j = E$
- 当 $\text{ce}_j > 1.0$（过载）时，$\text{fi}_j > E$  
- 当 $\text{ce}_j < 1.0$（闲置）时，$\text{fi}_j < E$

### 关键不对称性来源

**1. $Pi_j$ 与 $ce_j$ 的相关性**

在训练过程中，$\text{Pi}_j$（专家得分）和 $\text{ce}_j$（使用频率）是正相关的：
- 如果一个专家被频繁使用（ce_j 大），门控网络会学习给它更高的得分（Pi_j 大）
- 如果一个专家很少被使用（ce_j 小），门控网络会学习给它更低的得分（Pi_j 小）

**2. 乘积效应的非线性**

损失是 $\text{Pi}_j \times \text{ce}_j$ 的乘积，这产生了非线性效应。

## 具体数值例子

### 均衡基准情况
```python
# 完美均衡
ce_balanced = [1.0, 1.0, 1.0, 1.0]    # 使用频率
Pi_balanced = [0.25, 0.25, 0.25, 0.25] # 专家得分
fi_balanced = [4.0, 4.0, 4.0, 4.0]     # 归一化频率

loss_balanced = sum(Pi_balanced[i] * fi_balanced[i] for i in range(4))
# = 0.25*4.0 * 4 = 4.0
```

### 不均衡情况分析

#### 情况1：轻微不均衡
```python
# 专家0,1轻微过载，专家2,3轻微闲置
ce_unbalanced = [1.2, 1.2, 0.8, 0.8]      # 使用频率
Pi_unbalanced = [0.3, 0.3, 0.2, 0.2]      # 专家得分（与使用频率相关！）
fi_unbalanced = [4.8, 4.8, 3.2, 3.2]      # 归一化频率

loss_unbalanced = (0.3*4.8 + 0.3*4.8 + 0.2*3.2 + 0.2*3.2)
# = 1.44 + 1.44 + 0.64 + 0.64 = 4.16
```

**分析：**
- 过载专家贡献：1.44 + 1.44 = 2.88（相比均衡时2.0，增加0.88）
- 闲置专家贡献：0.64 + 0.64 = 1.28（相比均衡时2.0，减少0.72）
- **净增加：0.88 - 0.72 = 0.16**

#### 情况2：严重不均衡
```python
# 专家0,1严重过载，专家2,3严重闲置
ce_unbalanced = [1.8, 1.8, 0.2, 0.2]      # 使用频率
Pi_unbalanced = [0.45, 0.45, 0.05, 0.05]  # 专家得分（相关性更强！）
fi_unbalanced = [7.2, 7.2, 0.8, 0.8]      # 归一化频率

loss_unbalanced = (0.45*7.2 + 0.45*7.2 + 0.05*0.8 + 0.05*0.8)
# = 3.24 + 3.24 + 0.04 + 0.04 = 6.56
```

**分析：**
- 过载专家贡献：3.24 + 3.24 = 6.48（相比均衡时2.0，增加4.48）
- 闲置专家贡献：0.04 + 0.04 = 0.08（相比均衡时2.0，减少1.92）
- **净增加：4.48 - 1.92 = 2.56**

## 核心不对称性来源

### 1. Pi 与 ce 的正相关性
这是最关键的因素！在现实中：

```python
# 过载专家的 Pi 会增加
ce_overload = 2.0    # 使用率是理想值的2倍
Pi_overload = 0.4    # 得分也相应提高（原本是0.25）

# 闲置专家的 Pi 会减少  
ce_idle = 0.5        # 使用率是理想值的一半
Pi_idle = 0.1        # 得分也相应降低（原本是0.25）
```

### 2. 乘积的非线性效应
损失计算是 $\text{Pi} \times \text{ce}$，这产生了凸函数特性：

```python
def contribution(Pi, ce):
    return Pi * ce * n_experts

# 当过载时，Pi和ce都增加，乘积增长更快
contribution_overload = 0.4 * 2.0 * 4 = 3.2  # 大幅增加

# 当闲置时，Pi和ce都减少，乘积衰减更快  
contribution_idle = 0.1 * 0.5 * 4 = 0.2      # 大幅减少
```

## 几何直观解释

### 损失曲面分析
想象一个二维情况（只有2个专家）：

```python
# 专家1的贡献：P1 × ce1 × 2
# 专家2的贡献：P2 × ce2 × 2
# 约束：P1 + P2 = 1, ce1 + ce2 = 2（因为平均使用率是1）
```

这个损失曲面是**凸的**，在均衡点 (P1=0.5, ce1=1.0) 处取得最小值。

### 梯度不对称性
从均衡点出发：
- 向过载方向移动：损失快速上升
- 向闲置方向移动：损失缓慢下降（因为有下限0）

## 实际训练动态

### 门控网络的学习行为
```python
# 训练过程中，门控网络同时学习：
# 1. 根据输入选择合适专家（Pi）
# 2. 响应aux_loss信号调整路由

# 当过载发生时：
# - 该专家的Pi值因为频繁使用而提高
# - 但aux_loss惩罚会推动降低Pi值
# - 这种竞争产生了稳定的均衡点
```

### 损失不对称性的重要性
如果损失是对称的（过载增加 = 闲置减少），那么：
```python
# 假设对称情况：
过载专家损失增加: +1.0
闲置专家损失减少: -1.0  
净效果: 0 → 模型没有动力改变！

# 实际不对称情况：
过载专家损失增加: +2.0
闲置专家损失减少: -1.0
净效果: +1.0 → 模型有动力重新均衡！
```

## 数学证明

对于凸函数 $f(x) = x^2$（我们的损失函数有类似性质）：
```python
# 从均衡点 x=1 出发
f(1.5) = 2.25    # 增加1.25
f(0.5) = 0.25    # 减少0.75
净变化: +0.5
```

这正是我们观察到的：向两个方向的偏离，增加量 > 减少量。

## 总结

过载带来的损失增加大于闲置带来的损失减少，主要原因：

1. **Pi与ce的正相关性**：过载专家的得分提高，进一步放大损失
2. **乘积的非线性**：Pi × ce 的凸函数特性
3. **下限约束**：闲置损失有下限（≥0），而过载损失无上限
4. **训练动态**：这种不对称性为模型提供了明确的优化方向

这种精心设计的不对称性确保了MoE系统能够有效地学习到负载均衡的路由策略，是所有MoE模型成功的关键因素之一。

# 强化学习

## RLHF - 基于人类反馈的强化学习
- 通过人类对模型输出的偏好进行评价来训练模型，使其生成更符合人类价值观和偏好的内容
- 优点：
    - 更贴近真实人类偏好
- 缺点：
    - 成本高
    - 效率低

## RLAIF - 基于AI反馈的强化学习
- 使用AI模型（通常是预训练的语言奖励模型）来提供反馈，而不直接依赖人类的人工标注
- 优点：
    - 自动化
    - 可扩展性强
- 缺点：
    - 可能偏离人类真实偏好

- RLHF 和 RLAIF 除了反馈的来源不同，其他并无任何区别

## Policy Optimization (PO)
- 优化期望：训练时，只需**最小化负目标函数**，即: $\mathcal{L_{PO}}=-\mathcal{J_{PO}}$
$\mathcal{J}_{PO} = \mathbb{E}_{q \sim P(Q), o \sim \pi(O|q)} \left[ \underbrace{f(r_t)}_{\text{策略项}} \cdot \underbrace{g(A_t)}_{\text{优势项}} - \underbrace{h(\text{KL}_t)}_{\text{正则项}} \right]$

    - 问题/提示词  $q$: 从数据集 $P(Q)$ 中采样
    - 模型输出序列 $o$: 由策略 $\pi$ 生成 
    - 策略项 $f(r_t)$: 告诉模型新旧策略偏差有多大，是否探索到了更好的token; 如何使用概率比 $r_t$? 
        - $r_t = \frac{\pi_\theta(o_t\|q, o_{<t})}{\pi_{ref}(o_t\|q, o_{<t})}$ $(0, +\infty)$
    - 优势项 $g(A_t)$: 衡量某个动作相比基线有多好; 如何计算优势 $A_t$
    - 正则项 $h(\text{KL}_t)$: 既防止跑偏又防止管的太死, 防止策略偏离参考模型太远; 如何约束变化幅度 $\text{KL}_t$

### Direct Preference Optimization - 直接偏好优化算法（DPO）
- 直接最大化"chosen优于rejected"的对数几率
- 无需同步训练Reward/Value模型
- DPO只需跑actor与ref两个模型，显存占用低、收敛稳定、实现简单
- off‑policy：使用静态偏好数据集，可反复多轮epoch
    - Ref模型固定（预先缓存输出）
- 局限在于不做在线探索，更多用于"偏好/安全"的人类价值对齐
- 对"能不能做对题"的智力能力提升有限（当然这也取决于数据集，大规模收集正反样本并人类评估很困难）
- 损失函数: 
    $\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)\right]$
    - 策略项: $f(r_t) = \log r_w - \log r_l$
        - 对比chosen vs rejected的概率比
    - 优势项: $g(A_t)$ = 无
        - 通过偏好对比，无需显式计算优势
    - 正则项: $h(\text{KL}_t)$ = 隐含在 $\beta$ 中
        - 控制偏离参考模型程度


