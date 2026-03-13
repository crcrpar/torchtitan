# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph-based paged SAC passes.

- apply_paged_sac_pass: Joint graph pass that marks nodes annotated with
  ``{"paged_stash": True}`` (via ``annotate_fn``) as MUST_SAVE +
  should_paged_stash=True.  Designed to compose with apply_sac_pass which
  handles the standard SAC annotations.
- apply_sac_grouped_mm_pass: Variant of apply_sac that also saves _grouped_mm
  outputs. Used as a baseline to compare regular saves vs paged stash saves,
  isolating the fragmentation benefit.
"""

import torch
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.passes import DEFAULT_SAC_SAVE_OPS, apply_sac_pass
from torchtitan.tools.logging import logger

# Extend the default SAC save ops with _grouped_mm for fair baseline comparison.
_SAC_SAVE_OPS_WITH_GROUPED_MM = DEFAULT_SAC_SAVE_OPS | {
    torch.ops.aten._grouped_mm.default,
}


def apply_sac_grouped_mm_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Apply SAC with _grouped_mm added to the save list.

    Same as ``apply_sac_pass`` but also marks ``_grouped_mm`` outputs as
    MUST_SAVE.  This provides a fair baseline for comparing regular tensor
    saves (which fragment the allocator due to dynamic MoE token counts)
    against paged stash saves.

    Use ``--compile.joint_passes apply_sac_grouped_mm`` for the baseline.
    Compare against ``--compile.joint_passes apply_sac apply_paged_sac``
    for the paged stash variant.
    """
    return apply_sac_pass(gm, op_list_to_save=_SAC_SAVE_OPS_WITH_GROUPED_MM)


def apply_paged_sac_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Annotate paged stash ops in the joint graph.

    Scans ALL nodes (including those after the output_node boundary) and marks
    nodes that carry the ``{"paged_stash": True}`` FX annotation with
    ``MUST_SAVE`` + ``should_paged_stash=True``.

    Nodes are identified by the ``"custom"`` metadata key set by
    ``torch.fx.traceback.annotate_fn({"paged_stash": True})``, which is
    applied to ``_run_experts_grouped_mm`` during parallelization.

    This pass is designed to compose with ``apply_sac_pass`` which handles the
    standard SAC annotations (save/recompute decisions for mm, attention, etc.).
    Use ``--compile.joint_passes apply_sac apply_paged_sac`` to run both.

    Args:
        gm: The joint forward-backward graph module.

    Returns:
        The annotated graph module.
    """
    paged_stash_count = 0

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.meta.get("custom", {}).get("paged_stash", False):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            node.meta["should_paged_stash"] = True
            node.meta["ac_graph_id"] = 0
            paged_stash_count += 1

    gm.recompile()
    logger.info(
        f"Applied paged SAC graph pass ({paged_stash_count} marked for paged stash)"
    )
    return gm
