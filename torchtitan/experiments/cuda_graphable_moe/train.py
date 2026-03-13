# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer for paged stash experiments.

Extends GraphTrainer to reset paged stash buffers at each training step.
"""

from dataclasses import dataclass, field
from typing import Iterator

import torch

from torchtitan.experiments.cuda_graphable_moe.configs import (
    PagedStashActivationCheckpointConfig,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


class PagedStashTrainer(GraphTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config):
        activation_checkpoint: PagedStashActivationCheckpointConfig = field(
            default_factory=PagedStashActivationCheckpointConfig
        )
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        # Reset paged stash buffers before each training step
        for model_part in self.model_parts:
            buffers = getattr(model_part, "_paged_stash_buffers", None)
            if buffers:
                for buf in buffers:
                    buf.reset()
            # Reset overflow flag
            overflow = getattr(model_part, "_paged_stash_overflow", None)
            if overflow is not None:
                overflow.zero_()

        super().train_step(data_iterator)

        # Check for overflow after the step (matching Megatron's overflow assertion)
        for model_part in self.model_parts:
            overflow = getattr(model_part, "_paged_stash_overflow", None)
            if overflow is not None and overflow.item() != 0:
                raise RuntimeError(
                    "PagedStashBuffer overflow detected! Increase "
                    "paged_stash_buffer_size_factor in activation_checkpoint config."
                )
