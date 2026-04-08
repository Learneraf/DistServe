class ModelTypes:
    opt_13b = 'OPT-13B'
    opt_66b = 'OPT-66B'
    opt_175b = 'OPT-175B'
    llama_7b = 'LLAMA-7B'
    llama_2_7b = 'LLAMA-2-7B'
    llama_3_2_1b = 'LLAMA-3.2-1B'
    llama_3_2_3b = 'LLAMA-3.2-3B'
    llama_3_1_8b = 'LLAMA-3.1-8B'


    @staticmethod
    def formalize_model_name(x):
        if x == ModelTypes.opt_13b:
            return 'facebook/opt-13b'
        if x == ModelTypes.opt_66b:
            return 'facebook/opt-66b'
        if x == ModelTypes.opt_175b:
            return 'facebook/opt-175b'

        if x == ModelTypes.llama_7b:
            return 'huggyllama/llama-7b'
        if x == ModelTypes.llama_2_7b:
            return 'anonymous4chan/llama-2-7b'

        if x == ModelTypes.llama_3_2_1b:
            return '/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2'
        if x == ModelTypes.llama_3_2_3b:
            return '/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2'
        if x == ModelTypes.llama_3_1_8b:
            return '/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2'

        raise ValueError(x)

    @staticmethod
    def model_str_to_object(model):
        if model == 'opt_13b' or model == "facebook/opt-13b":
            return ModelTypes.opt_13b
        if model == 'opt_66b' or model == "facebook/opt-66b":
            return ModelTypes.opt_66b

        if model == 'huggyllama/llama-7b':
            return ModelTypes.llama_7b
        if model == 'anonymous4chan/llama-2-7b':
            return ModelTypes.llama_2_7b
        
        if model == '/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2':
            return ModelTypes.llama_3_2_1b
        if model == '/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2':
            return ModelTypes.llama_3_2_3b
        if model == '/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2':
            return ModelTypes.llama_3_1_8b
            
        if model == 'opt_175b' or model == "facebook/opt-175b":
            return ModelTypes.opt_175b
        raise ValueError(model)
