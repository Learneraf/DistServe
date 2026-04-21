import dataclasses
from typing import List
import marshal
import json

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    reqs: List[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "reqs": [(req.prompt, req.prompt_len, req.output_len) for req in self.reqs]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        try:
            with open(input_path, "rb") as f:
                loaded_data = marshal.load(f)
            return Dataset(
                loaded_data["dataset_name"],
                [TestRequest(req[0], req[1], req[2]) for req in loaded_data["reqs"]]
            )
        except (ValueError, EOFError, TypeError):
            pass

        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()

        if not raw_text:
            raise ValueError(f"Dataset file is empty: {input_path}")

        try:
            if raw_text[0] == "[":
                items = json.loads(raw_text)
            else:
                items = [
                    json.loads(line)
                    for line in raw_text.splitlines()
                    if line.strip()
                ]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Unsupported dataset format for {input_path}. "
                "Expected a marshal .ds file or JSON/JSONL records with "
                "`prompt`, `prompt_len`, and `output_len`/`output_tokens`."
            ) from exc

        reqs: List[TestRequest] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Unsupported JSON dataset item in {input_path}: {type(item).__name__}"
                )

            prompt = item.get("prompt")
            prompt_len = item.get("prompt_len")
            output_len = item.get("output_len", item.get("output_tokens"))
            if prompt is None or prompt_len is None or output_len is None:
                raise ValueError(
                    f"JSON dataset item in {input_path} is missing required keys. "
                    "Expected `prompt`, `prompt_len`, and `output_len`/`output_tokens`."
                )
            reqs.append(TestRequest(prompt, int(prompt_len), int(output_len)))

        return Dataset("jsonl", reqs)
        
import dataclasses
import numpy as np
from typing import List

from distserve.lifetime import LifetimeEvent, LifetimeEventType, json_decode_lifetime_events

class RequestResult:
    """
    A class for storing the results of a single request
    """
    
    def __init__(
        self,
        prompt_len: int,
        output_len: int,
        start_time: float,
        end_time: float,
        token_timestamps: List[float],
        lifetime_events: List[LifetimeEvent] = None
    ):
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.start_time = start_time
        self.end_time = end_time
        self.token_timestamps = token_timestamps
        self.lifecycle_events = lifetime_events
        
        self.latency = end_time - start_time
        self.ftl = token_timestamps[0] - start_time
        self.tpot = 0 if output_len == 1 else (token_timestamps[-1] - token_timestamps[0]) / (output_len-1)

def read_request_results(path: str) -> List[RequestResult]:
    with open(path, "r") as f:
        request_results: List[RequestResult] = [
            RequestResult(
                item["prompt_len"],
                item["output_len"],
                item["start_time"],
                item["end_time"],
                item["token_timestamps"],
                json_decode_lifetime_events(item["lifecycle_events"]) if item.get("lifecycle_events", None) is not None else None
            )
            for item in json.load(f)
        ]
    return request_results

def count_valid_results(request_results: list[RequestResult], ftl: float, tpot: float) -> int:
    """
    count_valid_results: Count the number of requests that satisfy the given FTL and TPOT.
    """
    count = 0
    for req in request_results:
        if req.ftl <= ftl and req.tpot <= tpot:
            count += 1
    return count

def get_slo_attainment(request_results: list[RequestResult], ftl: float, tpot: float) -> float:
    """
    get_slo_attainment: Get the SLO attainment of the given request results under the given FTL and TPOT.
    """
    return count_valid_results(request_results, ftl, tpot) / len(request_results)
