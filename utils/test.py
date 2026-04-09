import torch
from transformers import AutoConfig, AutoModel

GB = 1 << 30

def get_model_params_from_config(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with torch.device('meta'):
        model = AutoModel.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# 示例
params = get_model_params_from_config("/users/rh/.cache/modelscope/hub/models/skyline2006/llama-7b")
print(f"Total parameters: {params * 2 / GB:,}")  # 输出: 6,738,415,616（与官方一致）