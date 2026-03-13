# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.cuda_graphable_moe.configs import (
    PagedStashActivationCheckpointConfig,
    to_paged_stash_config,
)
from torchtitan.experiments.cuda_graphable_moe.train import PagedStashTrainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_16b,
    deepseek_v3_671b,
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_flex_attn,
)

from . import model_registry


def _apply_paged_stash_defaults(config: PagedStashTrainer.Config) -> None:
    """Apply paged stash defaults: hybridep backend, paged SAC, and cudagraph.

    Paged stash for DeepSeek V3 requires HybridEP to make MoE dispatch/combine
    CUDA-graph compatible. The default configuration enables:
    - HybridEP non-blocking dispatch (capacity_factor=1.0)
    - Graph-based paged SAC as the joint pass
    - CUDAGraph as the compiler pass
    """
    config.activation_checkpoint = PagedStashActivationCheckpointConfig()
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        joint_passes=["apply_sac", "apply_paged_sac"],
        passes=["cudagraph"],
    )
    config.parallelism.expert_parallel_comm_backend = "hybridep"
    config.parallelism.hybridep_non_blocking_expert_capacity_factor = 1.0


def paged_stash_deepseek_v3_debugmodel() -> PagedStashTrainer.Config:
    config = to_paged_stash_config(deepseek_v3_debugmodel(), model_registry)
    _apply_paged_stash_defaults(config)
    return config


def paged_stash_deepseek_v3_debugmodel_flex_attn() -> PagedStashTrainer.Config:
    config = to_paged_stash_config(
        deepseek_v3_debugmodel_flex_attn(), model_registry
    )
    _apply_paged_stash_defaults(config)
    return config


def paged_stash_deepseek_v3_16b() -> PagedStashTrainer.Config:
    config = to_paged_stash_config(deepseek_v3_16b(), model_registry)
    _apply_paged_stash_defaults(config)
    return config


def paged_stash_deepseek_v3_671b() -> PagedStashTrainer.Config:
    config = to_paged_stash_config(deepseek_v3_671b(), model_registry)
    _apply_paged_stash_defaults(config)
    return config
