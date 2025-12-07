import torch
from torch import optim, nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        """
        矩阵A高斯初始化
        引入随机性，打破对称性
        如果A也是全零初始化，那么ΔW = B×A = 0,无法开始学习
        标准差0.02：这是Transformer模型中常用的初始化标准差
        """
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        """
        矩阵B全0初始化
        确保训练开始时ΔW = 0
            保持预训练知识：训练开始时，模型完全保留预训练权重W₀
            稳定训练：避免初始阶段对下游任务的破坏性干扰
            渐进式适应：随着训练进行，B逐渐学习如何从A的表示中重建有用的更新
        """
        self.B.weight.data.zero_()
        
    def forward(self, x):
        return self.B(self.A(x))
    
    
def apply_lora(model: nn.Module, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            """
            只对方形矩阵（输入输出维度相同）的线性层应用LoRA
                Query/Key/Value投影层
                输出投影层
                跳过前馈网络（FFN）
            """
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(model.device)

            setattr(module, "lora", lora)
            original_forward = module.forward
            
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            
            # 显式绑定
            module.forward = forward_with_lora
            
    
def load_lora(model: nn.Module, path):
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {k.replace(f"{name}.lora.", ""):v for k, v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state)
            

def save_lora(model: nn.Module, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
            
    torch.save(state_dict, path)
            