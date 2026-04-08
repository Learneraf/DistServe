'''
usage:

python ./3-compare-time-usage.py \
    --std "./result/distserve-100-100.exp" \
    --sim "./result/sim-distserve-100-100.exp"
'''


import argparse

from structs import RequestResult as ReqResult, Dataset, read_request_results as load_req_result_list

def compare(
    std_reqs: list[ReqResult],
    sim_reqs: list[ReqResult]
):
    num_prompts = len(std_reqs)
    if num_prompts != len(sim_reqs):
        raise ValueError(f"Number of prompts in the standard and simulated results are different: {num_prompts} vs {len(sim_reqs)}")

    for i in range(num_prompts):
        std_req = std_reqs[i]
        sim_req = sim_reqs[i]
        if std_req.prompt_len != sim_req.prompt_len or std_req.output_len != sim_req.output_len:
            print(f"Prompt length or output length mismatch at index {i}: {std_req.prompt_len} vs {sim_req.prompt_len}, {std_req.output_len} vs {sim_req.output_len}")
            print("Falling back to sorting by prompt length and output length")
            std_reqs.sort(key=lambda x: (x.prompt_len, x.output_len))
            sim_reqs.sort(key=lambda x: (x.prompt_len, x.output_len))
            compare(std_reqs, sim_reqs)
            return
        print(f"{std_req.prompt_len:4d} {std_req.output_len:4d} {std_req.ftl:8.2f} {sim_req.ftl:8.2f} ({(sim_req.ftl-std_req.ftl)/std_req.ftl*100:5.1f}%) {std_req.tpot:8.2f} {sim_req.tpot:8.2f} {(sim_req.tpot-std_req.tpot)/std_req.tpot*100:5.1f} %")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--std", type=str, required=True, help="Path to the standard exp result")
    parser.add_argument("--sim", type=str, required=True, help="Path to the simulated exp result")
    args = parser.parse_args()
    
    std_reqs = load_req_result_list(args.std)
    sim_reqs = load_req_result_list(args.sim)
    
    compare(std_reqs, sim_reqs)
    