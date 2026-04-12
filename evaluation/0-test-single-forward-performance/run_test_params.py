import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from structs import *
from db import RecordManager
from sut.abstract_sut import SystemUnderTest
from sut.sut_distserve import DistServeSUT

def run_test_params(
    sut_class: type[SystemUnderTest],
    db_path: str,
    testing_params: list[TestParamGroup],
    warmup_rounds: int = 1,
    measure_rounds: int = 3,
    skip_duplicated: bool = True,
    store_into_db: bool = True,
    exp_output_dir: str | None = None,
):
    """
    Create a SUT, run it on a series of TestParamGroups, and store the results into a database
    """
    record_manager = RecordManager(db_path)
    exp_root = Path(exp_output_dir) if exp_output_dir is not None else None
    total_num_params = sum([len(test_param_group.input_params) for test_param_group in testing_params])
    num_finished_params = 0
    print(f"Total number of params to run: {total_num_params}")
    for test_param_group in testing_params:
        worker_param = test_param_group.worker_param
        # Check if we've already tested this set of parameters
        all_records_exist = True
        for input_param in test_param_group.input_params:
            if record_manager.query_record(worker_param, input_param) is None:
                all_records_exist = False
                break
        if all_records_exist and skip_duplicated:
            print(f"Record for test param group with {worker_param=} already exists. Continued")
            num_finished_params += len(test_param_group.input_params)
            continue

        print("==================================")
        print("==================================")
        print(f"Creating SUT with worker param {worker_param}")
        sut = sut_class(worker_param, test_param_group.input_params)

        for input_param in test_param_group.input_params:
            print("--------------------")
            print(f"Progress: {num_finished_params} / {total_num_params} ({num_finished_params/total_num_params*100:.2f}%)")
            print(f"Input param: {input_param}")
            if skip_duplicated and record_manager.query_record(worker_param, input_param) is not None:
                print("Skipped")
                num_finished_params += 1
                continue

            # Warm up the workers
            print(f"Warming up")
            for _ in range(warmup_rounds):
                sut.inference(input_param)

            # Benchmark
            print(f"Running")
            prefill_time_usages = []    # [measure_rounds]
            decoding_time_usages = []   # [measure_rounds*(output_len-1)]
            last_per_request_results = None
            for _ in range(measure_rounds):
                input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usage, per_request_results = sut.inference(input_param)
                prefill_time_usages.append(prefill_time_usage)
                decoding_time_usages.extend(decoding_time_usage)
                last_per_request_results = per_request_results
            avg_prefill_time_usage = np.mean(prefill_time_usages)
            avg_decoding_time_usage = np.median(decoding_time_usages)
            prefill_time_stddev = np.std(prefill_time_usages)
            decoding_time_stddev = np.std(decoding_time_usages)

            print(f"Pred output[0]: {predict_texts[0]}")
            print(f"Prefill time usage: avg {avg_prefill_time_usage}, stddev {prefill_time_stddev}")
            print(f"Decoding time usage: avg {avg_decoding_time_usage}, stddev {decoding_time_stddev}")
            # print(f"Prefill time usages: {prefill_time_usages}")
            # print(f"Decoding time usages: {decoding_time_usages}")
            if store_into_db:
                record_manager.update_or_insert_record(
                    worker_param,
                    input_param,
                    avg_prefill_time_usage,
                    avg_decoding_time_usage,
                    prefill_time_stddev,
                    decoding_time_stddev
                )
            if exp_root is not None and last_per_request_results is not None:
                model_dir_name = Path(worker_param.model_dir).parent.name
                exp_dir = exp_root / model_dir_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                exp_path = exp_dir / (
                    f"tp{worker_param.tp_world_size}"
                    f"-bs{input_param.batch_size}"
                    f"-in{input_param.input_len}"
                    f"-out{input_param.output_len}.exp"
                )
                with open(exp_path, "w", encoding="utf-8") as f:
                    json.dump([dataclasses.asdict(item) for item in last_per_request_results], f, indent=2)
                    f.write("\n")
            num_finished_params += 1
        del sut
        

if __name__ == "__main__":

    example_testing_params = [
        TestParamGroup(
            worker_param = WorkerParam(
                model_dir = "facebook/opt-13b",
                tp_world_size = 1,
                max_req_num = 1024,
                max_seq_len = 2048,
                use_dummy_weights = True
            ),
            input_params = [
                InputParam(
                    batch_size = 1,
                    input_len = 2000,
                    output_len = 16,
                )
            ]
        )
    ]

    run_test_params(
        DistServeSUT,
        "db-identical-req.sqlite",
        example_testing_params,
        warmup_rounds=2,
        measure_rounds=3,
        skip_duplicated=False,
        store_into_db=False
    )
