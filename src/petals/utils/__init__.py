from petals.utils.auto_config import (
    AutoDistributedConfig,
    AutoDistributedModel,
    AutoDistributedModelForCausalLM,
    AutoDistributedModelForSequenceClassification,
    ORTDistributedModel,
    ORTDistributedModelForCausalLM,
)
from petals.utils.dht import declare_active_modules, get_remote_module_infos
