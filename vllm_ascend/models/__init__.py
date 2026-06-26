from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")

    from vllm.model_executor.model_loader.weight_processing import (
        WeightProcessingFactory,
    )

    WeightProcessingFactory.register_module(
        "DSAAttention",
        "vllm_ascend.models.layer.attention.layer",
        "DSAAttention",
    )
