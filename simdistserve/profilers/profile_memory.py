import time
import csv
import os
import argparse
from config import ModelConfig, ParallelConfig

MB = 1 << 20
GB = 1 << 30

MODEL_LISTS = ["facebook/opt-13b", "facebook/opt-66b"]

try:
    import torch

    single_gpu_memory_ = torch.cuda.get_device_properties(0).total_memory
except:
    single_gpu_memory_ = 80 * GB
    pass


def _get_block_size_in_bytes(
    block_size: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    # the shape of one slot in k/v cache is [num_layers, num_local_heads, block_size, head_dim]
    num_layers = model_config.get_num_layers(parallel_config)
    num_heads = model_config.get_num_heads(parallel_config)
    head_dim = model_config.get_head_size()

    key_cache_size = num_layers * num_heads * block_size * head_dim
    total = key_cache_size * 2
    dtype_size = model_config.get_dtype_size()
    return total * dtype_size


def get_model_possible_pp(model):
    model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
    total_num_hidden_layers = model_config.hf_config.num_hidden_layers
    possible_pp = []
    for pp in range(1, 1 + total_num_hidden_layers):
        if total_num_hidden_layers % pp == 0:
            possible_pp.append(pp)
    return possible_pp


def get_model_possible_tp(model, num_gpus_per_node):
    model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
    total_num_attention_heads = model_config.hf_config.num_attention_heads
    possible_tp = []
    # 张量并行度不能超过每个节点下的GPU数量
    max_tp = min(total_num_attention_heads, num_gpus_per_node)
    for tp in range(1, 1 + max_tp):
        if total_num_attention_heads % tp == 0:
            possible_tp.append(tp)
    return possible_tp


def measure_stats(
    model="facebook/opt-13b", tp=1, pp=1,
    single_gpu_memory=single_gpu_memory_,
    block_size=16,
    gpu_memory_utilization=0.85,
    num_nodes=1,
    num_gpus_per_node=8,
):
    try:
        result = dict(locals())

        model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
        total_num_hidden_layers = model_config.hf_config.num_hidden_layers
        total_num_attention_heads = model_config.hf_config.num_attention_heads
        if total_num_hidden_layers % pp != 0:
            return None
        if total_num_attention_heads % tp != 0:
            return None

        # 计算总GPU数量和验证配置可行性
        total_gpus = num_nodes * num_gpus_per_node
        required_gpus = tp * pp
        if required_gpus > total_gpus:
            return None
        # 张量并行必须在单个节点内完成
        if tp > num_gpus_per_node:
            return None

        parallel_config = ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp
        )
        model_in_bytes_per_gpu = model_config.get_model_size_in_bytes(
            parallel_config=parallel_config
        )
        model_size_per_gpu = model_in_bytes_per_gpu / GB
        block_size_in_bytes = _get_block_size_in_bytes(
            block_size, model_config, parallel_config
        )

        # 计算更准确的运行时内存峰值
        # 1. 固定开销（操作系统、CUDA上下文等）
        fixed_overhead = single_gpu_memory * 0.15
        # 2. 模型本身的内存
        model_memory = model_in_bytes_per_gpu
        # 3. 中间激活值的内存（估计为模型大小的一定比例）
        activation_memory = model_in_bytes_per_gpu * 0.5
        # 4. 初始KV缓存的内存（考虑一个最小批处理大小）
        min_batch_size = 1
        initial_kv_memory = block_size_in_bytes * min_batch_size
        
        single_gpu_peak_runtime_memory = (
            fixed_overhead
            + model_memory
            + activation_memory
            + initial_kv_memory
        )
        num_gpu_blocks = int(
            (single_gpu_memory * gpu_memory_utilization - single_gpu_peak_runtime_memory)
            // block_size_in_bytes
        ) * total_gpus

        if num_gpu_blocks < 0:
            return None

        max_num_tokens = num_gpu_blocks * block_size
        kv_size_in_byte = (block_size_in_bytes / block_size / MB)
        result.update(
            dict(
                single_gpu_memory=single_gpu_memory,
                model_size_per_gpu=model_size_per_gpu,
                single_gpu_peak_runtime_memory=single_gpu_peak_runtime_memory,
                kv_size_in_byte=kv_size_in_byte,
                num_gpu_blocks=num_gpu_blocks,
                max_num_tokens=max_num_tokens,
                total_gpus=total_gpus,
                required_gpus=required_gpus,
            )
        )
        return result
    except Exception as e:
        # print(f"Error {model} {tp} {pp}: {e}")
        pass
    return None


def get_all_possible_tp_pp(num_gpus_per_node=8):
    for model in MODEL_LISTS:
        tps = get_model_possible_tp(model, num_gpus_per_node)
        pps = get_model_possible_pp(model)
        print((model, tps, pps))


def get_all_configs(output_file="profile_result.csv", num_nodes=1, num_gpus_per_node=8,
                    single_gpu_memory=single_gpu_memory_, block_size=16,
                    gpu_memory_utilization=0.85):
    configs = []
    fieldnames = None

    for model in MODEL_LISTS:
        for tp in get_model_possible_tp(model, num_gpus_per_node):
            for pp in get_model_possible_pp(model):
                # 计算该tp/pp组合需要的GPU数量
                required_gpus = tp * pp
                total_gpus = num_nodes * num_gpus_per_node
                if required_gpus > total_gpus:
                    continue
                a = measure_stats(model, tp, pp, single_gpu_memory, 
                                block_size, gpu_memory_utilization,
                                num_nodes, num_gpus_per_node)
                print(model, tp, pp, f"nodes={num_nodes}, gpus_per_node={num_gpus_per_node}, required_gpus={required_gpus}")
                if a is not None:
                    configs.append(a)
                    # 记录字段名
                    if fieldnames is None:
                        fieldnames = list(a.keys())

    # 将结果写入CSV文件
    if configs and fieldnames:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(configs)
        print(f"Save configs to {output_file}, total {len(configs)} records")

    return configs


def get_params(model):
    model_config = ModelConfig(model=model, tokenizer="facebook/opt-1.3b")
    total_num_hidden_layers = model_config.hf_config.num_hidden_layers
    total_num_attention_heads = model_config.hf_config.num_attention_heads
    return total_num_hidden_layers, total_num_attention_heads


def parse_args():
    parser = argparse.ArgumentParser(description="模型配置分析工具")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="num nodes")
    parser.add_argument("--num-gpus-per-node", type=int, default=2,
                        help="num gpus per node")
    parser.add_argument("--single-gpu-memory", type=int, default=None,
                        help="single gpu memory (bytes)")
    parser.add_argument("--block-size", type=int, default=16,
                        help="block size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95,
                        help="gpu memory utilization")
    parser.add_argument("--output-file", type=str, default="profile_result.csv",
                        help="output file path")
    parser.add_argument("--model-list", type=str, nargs="+", default=None,
                        help="model list to analyze")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置GPU内存
    single_gpu_memory = args.single_gpu_memory if args.single_gpu_memory else single_gpu_memory_
    
    # 设置模型列表
    model_lists = args.model_list if args.model_list else MODEL_LISTS
    
    print(f"Config: num_nodes={args.num_nodes}, num_gpus_per_node={args.num_gpus_per_node}, "
          f"total_gpus={args.num_nodes * args.num_gpus_per_node}")
    print(f"gpu memory: {single_gpu_memory / GB:.2f} GB")
    
    configs = get_all_configs(
        output_file=args.output_file,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        single_gpu_memory=single_gpu_memory,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )


if __name__ == '__main__':
    main()

    # usage: 
    # python profile_memory.py