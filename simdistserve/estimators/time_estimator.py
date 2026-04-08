# Fit a model where prefill does not have an intercept, and decode does have one.
import json
from pathlib import Path

from simdistserve.constants import ModelTypes


def load_distserve_profile_data():
    profile_data_path = Path(__file__).parent / "profiled_data" / "vllm-ascend" / "fit_params_llama_1B_num_prompt_100_5_params_decode.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


def load_vllm_profile_data():
    profile_data_path = Path(__file__).parent / "profiled_data" / "profiler-a100-80g.vllm.json"
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


distserve_profile_data = load_distserve_profile_data()
vllm_profile_data = load_vllm_profile_data()


def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, engine_type="distserve", **kw):
    if engine_type == "distserve":
        params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        a, b, c = params["prefill"]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        a, b, c = params["prefill"]

    f = 1
    a, b, c = (a * f, b * f, c * f)
    pp_factor = 1 / pp
    pp_const = 1 * pp  # TODO: Modulate the PP overhead
    num_total_tokens = sum(prefill_len_list)
    sum_num_tokens_sqr = sum([x ** 2 for x in prefill_len_list])
    delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
    delay = delay * pp_factor + pp_const
    return delay


# def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1, token_generated_list=None,
#                     engine_type="distserve", **kw):
#     batch_size = num_requests
#     if engine_type == "distserve":
#         params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
#         threshold = params[
#             "decoding_large_small_bs_threshold"]
#         if batch_size < threshold:
#             a, b, c = params["decoding_smallbs"]
#         else:
#             a, b, c = params["decoding_largebs"]
#     else:
#         params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
#         threshold = params[
#             "decoding_large_small_bs_threshold"]
#         if batch_size < threshold:
#             a, b, c = params["decoding_smallbs"]
#         else:
#             a, b, c = params["decoding_largebs"]
#         pass
#     f = 1
#     pp_factor = 1 / pp
#     # pp_const = 1 * pp  # TODO: Modulate the PP overhead
#     pp_const = 0
#     num_total_tokens = sum(token_generated_list)

#     delay = a + b * num_total_tokens + c * batch_size
#     delay = delay * pp_factor + pp_const
#     delay *= f
#     return delay


def get_decode_time(num_requests, pp=1, model_type=ModelTypes.opt_13b, TP=1,
                    token_generated_list=None, input_lens=None,  # 新增 input_lens
                    engine_type="distserve", **kw):
    batch_size = num_requests
    if engine_type == "distserve":
        params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params["decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            coeffs = params["decoding_smallbs"]   # 可能是3个或5个系数
        else:
            coeffs = params["decoding_largebs"]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params["decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            coeffs = params["decoding_smallbs"]
        else:
            coeffs = params["decoding_largebs"]

    # 判断系数数量，选择模型
    if len(coeffs) == 3:
        # 原三系数模型：A + B * total_output_tokens + C * batch_size
        a, b, c = coeffs
        num_total_tokens = sum(token_generated_list) if token_generated_list else 0
        delay = a + b * num_total_tokens + c * batch_size
    elif len(coeffs) == 5:
        # 五系数模型（方案三）：A + B*input_len + C*output_len + D*input_len*output_len + E*output_len^2
        # 需要逐个请求计算，然后求和？还是计算总延迟？
        # 注意：原函数返回的是单个请求的延迟还是整个批次的延迟？从原公式看，a+b*总token+c*batch_size 返回的是批次总延迟。
        # 五系数模型通常也是计算批次中所有请求的总解码时间。
        A, B, C, D, E = coeffs
        if input_lens is None or token_generated_list is None:
            raise ValueError("For 5-coefficient model, both input_lens and token_generated_list must be provided.")
        total_delay = 0.0
        for in_len, total_len in zip(input_lens, token_generated_list):
            # 单个请求的解码时间
            out_len = total_len - in_len
            req_delay = (A +
                         B * in_len +
                         C * out_len +
                         D * in_len * out_len +
                         E * out_len * out_len)
            total_delay += req_delay
        delay = total_delay
    else:
        raise ValueError(f"Unsupported number of coefficients: {len(coeffs)}")

    f = 1
    pp_factor = 1 / pp
    pp_const = 0
    delay = delay * pp_factor + pp_const
    delay *= f
    return delay