import json


def load_config():
    config_path = "src/utils/model_path.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_model(model_name, **kwargs):
    config = load_config()
    if model_name in config:
        model_path = config[model_name]
        
    if model_name.startswith("qwen2"):
        from src.models.qwen_model import QwenModel
        return QwenModel(model_name, model_path, **kwargs)
    elif model_name.startswith("llava"):
        from src.models.llava_model import LLaVAModel
        return LLaVAModel(model_name, model_path, **kwargs)
    elif model_name == "qwen-vl-max":
        from src.models.api_model import APIModel
        return APIModel(api_name="qwen", model_name=model_name, **kwargs)
    elif model_name.startswith("gpt") or model_name.startswith("gemini"):
        from src.models.api_model import APIModel
        return APIModel(api_name="agicto", model_name=model_name, **kwargs)
    elif model_name == "qwen_32b":
        from src.models.api_model import APIModel
        return APIModel(api_name="local", model_name="Qwen2.5-32B-Instruct-AWQ", **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")