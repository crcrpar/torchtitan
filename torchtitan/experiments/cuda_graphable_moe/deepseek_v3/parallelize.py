# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelize DeepSeek V3 with graph_trainer AOT compilation and paged SAC.

Extends graph_trainer's parallelize_deepseekv3 with graph-based paged SAC
for MoE expert activations when ``apply_paged_sac`` is in joint_passes.

Usage:
    # Without paged stash (standard graph_trainer SAC):
    --compile.joint_passes apply_sac

    # With paged stash (apply_sac handles SAC annotations, apply_paged_sac
    # adds paged stash metadata on top):
    --compile.joint_passes apply_sac apply_paged_sac
"""

import torch.nn as nn

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as graph_trainer_parallelize_deepseekv3,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_deepseekv3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Parallelize DeepSeek V3 with graph_trainer + optional paged SAC.

    When ``apply_paged_sac`` is in ``compile_config.joint_passes``, this
    function allocates paged stash buffers (which require the parallelized
    model) before delegating compilation to ``apply_compile``.  Both
    ``apply_sac`` and ``apply_paged_sac`` are resolved from the standard
    pass registry.  Otherwise delegates entirely to graph_trainer's
    ``parallelize_deepseekv3``.
    """
    # Check if paged SAC is requested
    joint_pass_names = getattr(compile_config, "joint_passes", [])
    paged_sac_enabled = "apply_paged_sac" in joint_pass_names

    if not paged_sac_enabled:
        # No paged stash — use graph_trainer's parallelize directly
        return graph_trainer_parallelize_deepseekv3(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )

    # Paged SAC enabled — annotate MoE expert computation so the graph pass
    # can identify stash-eligible nodes by annotation rather than op target.
    # This must happen before graph_trainer_parallelize_deepseekv3 which
    # triggers annotate_deepseekv3() and eventually AOT tracing.
    from torch.fx.traceback import annotate_fn

    import torchtitan.models.common.moe.moe as moe_module

    moe_module._run_experts_grouped_mm = annotate_fn({"paged_stash": True})(
        moe_module._run_experts_grouped_mm
    )

    # We need to allocate paged buffers after parallelization but before
    # compilation.  Use graph_trainer's parallelize for TP/EP/DP/AC/hybridep
    # setup, but temporarily disable compilation so we can handle it ourselves.
    original_enable = compile_config.enable
    compile_config.enable = False
    model = graph_trainer_parallelize_deepseekv3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        model_converters=model_converters,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    compile_config.enable = original_enable

    # Set up paged stash buffers (requires parallelized model to scan
    # GroupedExperts modules for dtype/hidden_size)
    from ..paged_stash_ac import (
        create_paged_buffers,
        get_max_in_flight_microbatches,
    )

    num_experts = model.config.layer.moe.num_experts
    top_k = model.config.layer.moe.top_k
    base_tokens = training.local_batch_size * training.seq_len
    if (
        parallelism.expert_parallel_comm_backend == "hybridep"
        and parallelism.hybridep_non_blocking_expert_capacity_factor is not None
        and parallel_dims.ep_enabled
    ):
        ep_size = parallel_dims.ep
        num_local_experts = num_experts // ep_size
        cf = parallelism.hybridep_non_blocking_expert_capacity_factor
        max_tokens = int(
            base_tokens * ep_size * min(num_local_experts, top_k) * cf
        )
    else:
        max_tokens = base_tokens * top_k

    max_in_flight = get_max_in_flight_microbatches(
        parallelism, training.local_batch_size
    )
    buffers, overflow = create_paged_buffers(
        model, ac_config, max_tokens=max_tokens, max_in_flight=max_in_flight
    )

    if buffers is not None:
        model._paged_stash_buffers = list(buffers.values())
        model._paged_stash_overflow = overflow
        logger.info("Graph-based paged SAC enabled")

    # Apply compilation — apply_sac and apply_paged_sac are resolved from
    # the pass registry via compile_config.joint_passes
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
