import os
from typing import List, Optional, Union

from hivemind import get_logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention, BloomConfig
from transformers.models.bloom.configuration_bloom import BloomOnnxConfig

from optimum.configuration_utils import BaseConfig
from transformers.onnx import PatchingSpec

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.bloom_onnx.block import WrappedONNXBloomBlock

logger = get_logger(__name__)

class WrappedBloomOnnxConfig(BloomConfig, BloomOnnxConfig):
    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,  # TP rank used when training with megatron
        slow_but_exact=False,
        **kwargs,
    ):
        super().__init__(vocab_size, hidden_size, n_layer, n_head, layer_norm_epsilon, initializer_range, use_cache, bos_token_id, eos_token_id,
                                          apply_residual_connection_post_layernorm, hidden_dropout, attention_dropout, pretraining_tp, slow_but_exact, **kwargs)
        super().__init__(self)

class DistributedBloomONNXConfig(WrappedBloomOnnxConfig, ClientConfig, PTuneConfig, LMHeadConfig):
# class DistributedBloomONNXConfig(BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedONNXBloomBlock
    attn_class = BloomAttention
    block_prefix = "h"

    num_key_value_groups = 1

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        logger.info("Make sure you follow the BLOOM's terms of use: https://bit.ly/bloom-license")

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            # We need "-petals" for backward compatibility with Petals < 1.2.0
            dht_prefix = str(model_name_or_path) + "-petals"
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")
        return super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
