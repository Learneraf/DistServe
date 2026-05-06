import json
import math
import os
from pathlib import Path

from simdistserve.constants import ModelTypes


def load_distserve_profile_data():
    base_dir = Path(__file__).parent / "profiled_data" / "distserve-cuda"
    profile_data_path = os.environ.get("SIMDISTSERVE_DISTSERVE_PROFILE")
    if profile_data_path is None:
        preferred_paths = [
            base_dir / "fit_params_live_5p4d.json",
            base_dir / "fit_params_live.json",
            base_dir / "fit_params_live_decode.json",
            base_dir / "fit_params.json",
        ]
        profile_data_path = next(path for path in preferred_paths if path.exists())
    else:
        profile_data_path = Path(profile_data_path)
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


def load_vllm_profile_data():
    base_dir = Path(__file__).parent / "profiled_data" / "distserve-cuda"
    profile_data_path = os.environ.get("SIMDISTSERVE_VLLM_PROFILE")
    if profile_data_path is None:
        profile_data_path = base_dir / "fit_params_cuda_data_fit_5p4d_infer_batch.json"
    else:
        profile_data_path = Path(profile_data_path)
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


def load_vllm_ascend_profile_data():
    base_dir = Path(__file__).parent / "profiled_data" / "vllm-ascend"
    profile_data_path = os.environ.get("SIMDISTSERVE_VLLM_ASCEND_PROFILE")
    if profile_data_path is None:
        preferred_paths = [
            base_dir / "fit_params_live_5p4d_filtered.json",
            base_dir / "fit_params_all.json",
        ]
        profile_data_path = next(path for path in preferred_paths if path.exists())
    else:
        profile_data_path = Path(profile_data_path)
    with open(profile_data_path) as f:
        profile_data = json.load(f)
        return profile_data


distserve_profile_data = load_distserve_profile_data()
vllm_profile_data = load_vllm_profile_data()
vllm_ascend_profile_data = load_vllm_ascend_profile_data()

def get_prefill_time(num_tokens=None, pp=1, bs=1, decode_bs=0, model_type=ModelTypes.opt_13b, TP=1,
                     prefill_len_list=None, engine_type="distserve", **kw):
    if engine_type == "distserve":
        params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
    elif engine_type == "vllm_ascend":
        params = vllm_ascend_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
    else:
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
    coeffs = params["prefill"]
    prefill_scale = params.get("prefill_scale", 1.0)

    if prefill_len_list is None:
        prefill_len_list = []
    num_total_tokens = sum(prefill_len_list)
    sum_num_tokens_sqr = sum(x ** 2 for x in prefill_len_list)
    max_num_tokens = max(prefill_len_list) if prefill_len_list else 0

    if len(coeffs) == 3:
        a, b, c = coeffs
        f = 1
        a, b, c = (a * f, b * f, c * f)
        delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
    elif len(coeffs) == 5:
        # Live prefill-batch fit: batch runtime depends on the current batch shape.
        a, b, c, d, e = coeffs
        delay = a + b * bs + c * num_total_tokens + d * max_num_tokens + e * sum_num_tokens_sqr
    else:
        raise ValueError(f"Unsupported number of prefill coefficients: {len(coeffs)}")

    pp_factor = 1 / pp
    pp_const = 1 * pp  # TODO: Modulate the PP overhead
    delay *= prefill_scale
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
                    token_generated_list=None, input_lens=None, output_lens=None,
                    current_context_lens=None,
                    engine_type="distserve", **kw):
    batch_size = num_requests
    if engine_type == "distserve":
        params = distserve_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params["decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            coeffs = params["decoding_smallbs"]
            if not coeffs:
                coeffs = params["decoding_largebs"]
        else:
            coeffs = params["decoding_largebs"]
            if not coeffs:
                coeffs = params["decoding_smallbs"]
    elif engine_type == "vllm":
        params = vllm_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params["decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            coeffs = params["decoding_smallbs"]
            if not coeffs:
                coeffs = params["decoding_largebs"]
        else:
            coeffs = params["decoding_largebs"]
            if not coeffs:
                coeffs = params["decoding_smallbs"]
    else:
        params = vllm_ascend_profile_data[ModelTypes.formalize_model_name(model_type)][str(TP)]
        threshold = params["decoding_large_small_bs_threshold"]
        if batch_size < threshold:
            coeffs = params["decoding_smallbs"]
            if not coeffs:
                coeffs = params["decoding_largebs"]
        else:
            coeffs = params["decoding_largebs"]
            if not coeffs:
                coeffs = params["decoding_smallbs"]
    
    if len(coeffs) == 3:
        a, b, c = coeffs
        num_total_tokens = sum(token_generated_list) if token_generated_list else 0
        delay = a + b * num_total_tokens + c * batch_size
    elif len(coeffs) == 4:
        # Live decode-round fit: round time depends on current batch state.
        a, b, c, d = coeffs
        if input_lens is None or output_lens is None or current_context_lens is None:
            raise ValueError(
                "For 4-coefficient model, input_lens, output_lens, and current_context_lens are required."
            )
        sum_context_len = sum(current_context_lens)
        max_context_len = max(current_context_lens) if current_context_lens else 0
        delay = (
            a
            + b * batch_size
            + c * sum_context_len
            + d * max_context_len
        )
    elif len(coeffs) == 5:
        # vLLM Ascend live decode-round fit:
        #   round_ms = A + B * batch_size
        #            + C * sum_context_len
        #            + D * max_context_len
        #            + E * sum_remaining_output_tokens
        a, b, c, d, e = coeffs
        if input_lens is None or output_lens is None or current_context_lens is None:
            raise ValueError(
                "For 5-coefficient model, input_lens, output_lens, and current_context_lens are required."
            )
        sum_context_len = sum(current_context_lens)
        max_context_len = max(current_context_lens) if current_context_lens else 0
        sum_remaining_output_tokens = sum(
            max(int(input_len) + int(output_len) - int(context_len), 0)
            for input_len, output_len, context_len in zip(input_lens, output_lens, current_context_lens)
        )
        delay = (
            a
            + b * batch_size
            + c * sum_context_len
            + d * max_context_len
            + e * sum_remaining_output_tokens
        )
    else:
        raise ValueError(f"Unsupported number of coefficients: {len(coeffs)}")

    f = 1
    pp_factor = 1 / pp
    pp_const = 0
    delay = delay * pp_factor + pp_const
    delay *= f
    return delay
