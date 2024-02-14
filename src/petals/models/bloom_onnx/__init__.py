from petals.models.bloom_onnx.block import WrappedONNXBloomBlock
from petals.models.bloom_onnx.config import DistributedBloomONNXConfig
from petals.models.bloom_onnx.model import (
    DistributedBloomONNXForCausalLM,
    DistributedBloomONNXForSequenceClassification,
    DistributedBloomONNXModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedBloomONNXConfig,
    model=DistributedBloomONNXModel,
    model_for_causal_lm=DistributedBloomONNXForCausalLM,
    model_for_sequence_classification=DistributedBloomONNXForSequenceClassification,
)
