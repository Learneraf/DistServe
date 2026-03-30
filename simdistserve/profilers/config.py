from typing import Optional
import torch
from typing import List
from transformers import AutoConfig, AutoModel
from deprecated import deprecated

class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        tensor_parallel_size: number of tensor parallel groups.
        tensor_parallel_rank: rank in the tensor parallel group.
        pipeline_parallel_size: number of pipeline parallel groups.
        pipeline_parallel_rank: rank in the pipeline parallel group.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_rank: int = 0,
    ) -> None:
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self.use_parallel = self.world_size > 1

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.tensor_parallel_size,
                self.tensor_parallel_rank,
                self.pipeline_parallel_size,
                self.pipeline_parallel_rank,
            ]
        )

    def to_list(self) -> List[int]:
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
        ]

    def is_last_stage(self) -> bool:
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Model name or path.
        tokenizer: Tokenizer name or path.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
            Default to "auto".
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        dtype: Data type of the model. Default to "fp16".
        seed: Random seed for reproducing.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str],
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        dtype: str = "fp16",
        seed: int = 1,
        use_dummy_weights: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer if tokenizer else model
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.seed = seed
        self._verify_args()
        self.hf_config = self._get_hf_config()
        self.use_dummy_weights = use_dummy_weights

    def _verify_args(self):
        assert self.dtype in [
            "fp16",
            "fp32",
        ], f"dtype must be either 'fp16' or 'fp32'."

    def _get_hf_config(self):
        try:
            config = AutoConfig.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            print(e)
            raise ValueError(
                f"Failed to load the model config, please check the model name or path: {self.model}"
            )
        return config

    def get_dtype_size(self) -> int:
        if self.dtype == "fp16":
            return 2
        elif self.dtype == "fp32":
            return 4
        else:
            raise NotImplementedError(f"dtype {self.dtype} not supported")

    def get_torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPE_MAP[self.dtype]

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_ffn_inter_dim(self) -> int:
        """获取 FFN 中间层维度（不同模型命名不同）。"""
        model_type = self.hf_config.model_type
        if model_type == "opt":
            # OPT 使用 ffn_dim
            return self.hf_config.ffn_dim
        elif hasattr(self.hf_config, "intermediate_size"):
            # LLaMA、Falcon 等
            return self.hf_config.intermediate_size
        else:
            # 回退值（4倍隐藏层大小是常见设置）
            return self.get_hidden_size() * 4

    def get_q_heads(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        # For LLaMA-2:
        return (
            self.hf_config.num_attention_heads
                // parallel_config.tensor_parallel_size
        )

    def get_num_heads(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        new_decoder_arch_falcon = self.hf_config.model_type == "falcon" and getattr(
            self.hf_config, "new_decoder_architecture", False
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1

        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv // parallel_config.tensor_parallel_size

        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (
                self.hf_config.num_key_value_heads
                // parallel_config.tensor_parallel_size
            )

        # Normal case:
        total_num_attention_heads = self.hf_config.num_attention_heads
        assert total_num_attention_heads % parallel_config.tensor_parallel_size == 0, (
            f"Total number of attention heads ({total_num_attention_heads}) "
            f"must be divisible by the size of tensor parallel group "
            f"({parallel_config.tensor_parallel_size})."
        )
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_max_model_len(self) -> int:
        max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        for key in possible_keys:
            max_len_key = getattr(self.hf_config, key, None)
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        return max_model_len

    def get_num_layers(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        assert total_num_hidden_layers % parallel_config.pipeline_parallel_size == 0, (
            f"Number of layers ({total_num_hidden_layers}) must be divisible "
            f"by the size of pipeline parallel group "
            f"({parallel_config.pipeline_parallel_size})."
        )
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    @deprecated("Use auto_get_model_params instead")
    def get_total_params(self) -> int:
        """计算模型总参数量（不区分并行）。"""
        h = self.get_hidden_size()
        vocab_size = self.hf_config.vocab_size
        ffn_dim = self.get_ffn_inter_dim()
        num_layers = self.hf_config.num_hidden_layers
        # 嵌入层参数（词嵌入 + 位置嵌入，若有）
        embedding_params = vocab_size * h
        # 位置嵌入：有的模型有可学习的位置编码，如 GPT-2；有的没有（如 LLaMA 使用 RoPE）
        if hasattr(self.hf_config, "max_position_embeddings"):
            embedding_params += self.get_max_model_len() * h
        # 每个 Transformer 层的参数
        # 注意力：QKV 投影 + 输出投影
        attn_params = 4 * h * h  # QKV 和输出投影

        if self.hf_config.model_type == "llama":
            # 三个线性层
            ffn_params = 3 * h * ffn_dim
        else:
            # 默认两个线性层
            ffn_params = 2 * h * ffn_dim

        # 层归一化等偏置（若模型使用）
        layer_norm_params = 2 * h  # 两个 LayerNorm 的增益和偏置
        per_layer_params = attn_params + ffn_params + layer_norm_params
        # 总参数量
        total_params = embedding_params + num_layers * per_layer_params
        # 输出层（LM head），通常与词嵌入共享权重
        # 这里保守估计不共享，实际需根据模型配置判断
        if not getattr(self.hf_config, "tie_word_embeddings", False):
            total_params += vocab_size * h
        return total_params

    def auto_get_model_params(self) -> int:
        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        with torch.device('meta'):
            model = AutoModel.from_config(config)
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    def get_model_size_in_bytes(
        self, parallel_config: ParallelConfig = ParallelConfig()
    ) -> int:
        """返回每个 GPU 上的模型参数内存占用（字节）。"""
        # total_params = self.get_total_params()

        total_params = self.auto_get_model_params()
        dtype_size = self.get_dtype_size()
        # 参数在 TP 和 PP 下均匀分布
        return (total_params * dtype_size) // (parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size)


def _get_block_size_in_bytes(
    block_size: int,
    model_config: ModelConfig,
    parallel_config: ParallelConfig
) -> int:
    """
    计算单个 KV Cache block 的大小（字节）。
    block_size: 每个 block 存储的 token 数
    """
    # 每个 token 在每个 GPU 上的 KV Cache 大小
    num_layers_per_gpu = model_config.get_num_layers_per_stage(parallel_config)
    num_kv_heads_per_gpu = model_config.get_num_kv_heads(parallel_config)
    head_dim = model_config.get_head_size()
    dtype_size = model_config.get_dtype_size()
    per_token_kv_bytes = (
        num_layers_per_gpu
        * num_kv_heads_per_gpu
        * head_dim
        * 2  # key and value
        * dtype_size
    )
    return per_token_kv_bytes * block_size