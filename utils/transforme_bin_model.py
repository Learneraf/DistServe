#!/usr/bin/env python3
"""
强制保存为 .bin 格式，不使用 save_pretrained
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.expanduser("~/.cache/modelscope/hub/models/LLM-Research")
MODEL_NAMES = [
    "Llama-3___2-1B",
    "Llama-3___2-3B",
    "Meta-Llama-3___1-8B",
    "llama-2-7b"
]
DTYPE = torch.float16
DEVICE_MAP = "auto"

def convert_model(model_dir: str, output_dir: str):
    print(f"\n🔍 加载模型: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取 state_dict 并移回 CPU（节省显存）
    state_dict = model.state_dict()
    # 如果模型在多 GPU 上，需要先移到 CPU
    state_dict = {k: v.cpu() for k, v in state_dict.items()}
    
    # 保存为单个 .bin 文件（如果模型很大，可以分片，但一般 8B 以内单个文件可接受）
    bin_path = os.path.join(output_dir, "pytorch_model.bin")
    print(f"💾 保存 state_dict 到 {bin_path}")
    torch.save(state_dict, bin_path)
    
    # 保存分词器和配置文件（复用原来模型目录中的文件）
    tokenizer.save_pretrained(output_dir)
    # 复制 config.json 等配置文件
    import shutil
    for fname in ["config.json", "generation_config.json", "modeling_lib.py"]:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_dir)
    
    print(f"✅ 转换完成！输出目录: {output_dir}")

def main():
    for name in MODEL_NAMES:
        model_path = os.path.join(BASE_DIR, name)
        if not os.path.isdir(model_path):
            print(f"⚠️ 跳过 {name}: 目录不存在")
            continue
        output_path = os.path.join(model_path, "converted_bin_v2")  # 新目录避免覆盖
        if os.path.exists(output_path) and os.listdir(output_path):
            print(f"⏭️ 跳过 {name}: {output_path} 已存在")
            continue
        try:
            convert_model(model_path, output_path)
        except Exception as e:
            print(f"❌ 转换 {name} 失败: {e}")

if __name__ == "__main__":
    main()