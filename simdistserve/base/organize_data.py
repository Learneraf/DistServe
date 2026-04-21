from typing import TypedDict
import numpy as np
import pandas as pd


class Request_t(TypedDict):
    req_id: int
    source_index: int
    prefill_lens: int
    output_lens: int
    pass


class RequestLog_t(TypedDict):
    start_time: float
    end_time: float
    event_type: str
    req_id: int
    duration: float
    pass


class WorkerLog_t(TypedDict):
    start_time: float
    end_time: float
    event_type: str
    worker_id: int
    duration: float
    decode_batch_size: int
    prefill_batch: 'list[int]'
    decode_batch: 'list[int]'
    pass


class LatencyDist_t(TypedDict):
    first_token_latency: float
    decoding_latency: float
    tpot: float
    inv_tpot_ms: float
    inv_tpot_s: float
    pass


def organize_request_df(requests) -> 'DataFrame[Request_t]':
    """Describe the property of each request."""
    request_df = pd.DataFrame([
        {
            'req_id': r.req_id,
            'source_index': getattr(r, 'source_index', r.req_id),
            'prefill_lens': r.prefill_lens,
            'output_lens': r.output_lens,
            'first_token_prefill': getattr(r, 'first_token_prefill', False),
        }
        for r in requests
    ])
    return request_df


def transform_request_log_to_df(req: 'Request') -> 'DataFrame[RequestLog_t]':
    """
    Transform the request log into a DataFrame.
    :param req:
    :return: DataFrame
        req_id, start_time, end_time, duration, event_type
    """
    df = pd.DataFrame(req.log, columns=['start_time', 'event_type', 'worker_id'])
    df['req_id'] = req.req_id
    df['duration'] = df['start_time'].shift(-1) - df['start_time']
    df['duration'] = df['duration'].fillna(0)
    df['end_time'] = df['start_time'] + df['duration']
    return df


def organize_request_event_df(requests) -> 'DataFrame[RequestLog_t]':
    """Aggregate all request event logs into a single DataFrame."""
    request_event_df = pd.concat([
        transform_request_log_to_df(r)
        for r in requests
    ])
    return request_event_df


def transform_worker_log_to_df(worker: 'Worker') -> 'DataFrame[WorkerLog_t]':
    if not worker.log:
        return None
    df = pd.DataFrame(worker.log, columns=[
        'start_time', 'event_type', 'num_tokens', 'prefill_bs', 'decode_bs',
        'prefill_batch',
        'decode_batch'
    ])
    df['worker_id'] = worker.wid
    df['duration'] = df['start_time'].shift(-1) - df['start_time']
    df['duration'] = df['duration'].fillna(0)
    df['end_time'] = df['start_time'] + df['duration']
    return df


def organize_worker_event_df(cluster) -> 'DataFrame[WorkerLog_t]':
    """Aggregate all worker event logs into a single DataFrame."""
    worker_event_df = pd.concat([
        transform_worker_log_to_df(w)
        for w in cluster.get_all_workers()
    ])
    return worker_event_df


def calculate_per_request_latency(
    df: 'DataFrame[RequestLog_t]',
    output_lens: 'pd.Series' = None,
    first_token_prefill: 'pd.Series' = None,
) -> 'DataFrame[LatencyDist_t]':
    assert isinstance(output_lens, pd.Series) or output_lens is None, \
        f'output_lens must be a pd.Series, got {type(output_lens)}'
    assert isinstance(first_token_prefill, pd.Series) or first_token_prefill is None, \
        f'first_token_prefill must be a pd.Series, got {type(first_token_prefill)}'
    # First token latency: time between request init and the first decode round finishing.
    # Decoding latency: time between the first and the last decode rounds finishing,
    # matching the benchmark's TPOT convention.
    first_event = df[df.event_type == 'init'].groupby('req_id').start_time.min()
    first_decode_end = df[df.event_type == 'do_decode'].groupby('req_id').end_time.min()
    last_decode_end = df[df.event_type == 'do_decode'].groupby('req_id').end_time.max()
    finish_prefill_time = df[df.event_type == 'finish_prefill'].groupby('req_id').start_time.max()
    first_visible_event = df[df.event_type == 'first_token_visible'].groupby('req_id').start_time.min()
    last_event = df[df.event_type == 'exit_system'].groupby('req_id').end_time.max()

    first_token_ready = first_decode_end.reindex(first_event.index)
    first_token_ready = first_visible_event.reindex(first_event.index).combine_first(first_token_ready)
    if first_token_prefill is not None:
        # Backward compatibility for older traces that only set the
        # `first_token_prefill` flag without emitting `first_token_visible`.
        first_token_prefill = first_token_prefill.astype(bool)
        prefill_first_ready = finish_prefill_time.reindex(first_event.index)
        use_prefill_finish = first_token_prefill & first_visible_event.reindex(first_event.index).isna()
        first_token_ready = first_token_ready.where(~use_prefill_finish, prefill_first_ready)

    # Then, calculate the first token latency and decoding latency for each req_id
    first_token_latency = (first_token_ready - first_event).fillna(0)
    decoding_latency = (last_decode_end - first_token_ready).fillna(0)
    total_latency = last_event - first_event

    dist_df = pd.DataFrame({
        'first_token_latency': first_token_latency,
        'decoding_latency': decoding_latency,
        'total_latency': total_latency,
    })

    if output_lens is not None:
        # Match the benchmark definition:
        #   TPOT = (last_token_time - first_token_time) / (output_len - 1)
        decode_output_tokens = (output_lens - 1).clip(lower=1)
        tpot = decoding_latency.div(decode_output_tokens).replace([np.inf, - np.inf], 0)
        tpot = tpot.where(output_lens > 1, 0)
        dist_df['tpot'] = tpot
        dist_df['inv_tpot_ms'] = (1 / tpot).replace([np.inf, - np.inf], 0)
        dist_df['inv_tpot_s'] = (1000 / tpot).replace([np.inf, - np.inf], 0)

    return dist_df
