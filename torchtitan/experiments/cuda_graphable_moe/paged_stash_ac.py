# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Paged stash buffer and Triton kernels for MoE activation storage.

Instead of recomputing activations (standard AC), stores them in a pre-allocated
paged buffer managed by Triton kernels. This avoids recompute cost while reducing
memory fragmentation for MoE expert layers with dynamic token counts.

Contains:
  - Triton kernels: _paged_stash_copy_kernel, _paged_stash_pop_kernel
  - PagedStashBuffer: Pre-allocated paged memory pool
  - create_paged_buffers: Factory for creating buffers from model structure
"""

import torch
import triton
import triton.language as tl

from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _paged_stash_copy_kernel(
    src_ptr, dst_ptr, num_tokens_ptr, free_list_ptr,
    free_list_head_ptr, free_list_tail_ptr, free_list_capacity_ptr,
    page_record_ptr, overflow_ptr, new_free_list_head_ptr,
    PAGE_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    num_tokens = tl.load(num_tokens_ptr)
    free_list_head = tl.load(free_list_head_ptr)
    free_list_tail = tl.load(free_list_tail_ptr)
    free_list_capacity = tl.load(free_list_capacity_ptr)
    avail_pages = free_list_tail - free_list_head
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
    overflow_detected = avail_pages < required_pages
    if pid == 0 and overflow_detected:
        tl.store(overflow_ptr, 1)
    if overflow_detected:
        return
    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        free_list_idx = (free_list_head + page_slot) % free_list_capacity
        page_id = tl.load(free_list_ptr + free_list_idx)
        if token_in_page == 0:
            tl.store(page_record_ptr + page_slot, page_id)
        dst_token_idx = page_id * PAGE_SIZE + token_in_page
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        token_idx_i64 = token_idx.to(tl.int64)
        dst_token_idx_i64 = dst_token_idx.to(tl.int64)
        src_base = src_ptr + token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + dst_token_idx_i64 * HIDDEN_SIZE
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        token_idx += num_blocks
    if pid == 0:
        new_head = free_list_head + required_pages
        tl.store(new_free_list_head_ptr, new_head)


@triton.jit
def _paged_stash_pop_kernel(
    src_ptr, dst_ptr, num_tokens_ptr, page_record_ptr,
    free_list_ptr, free_list_head_ptr, free_list_tail_ptr,
    free_list_capacity_ptr, new_free_list_tail_ptr,
    PAGE_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    num_tokens = tl.load(num_tokens_ptr)
    free_list_tail = tl.load(free_list_tail_ptr)
    free_list_capacity = tl.load(free_list_capacity_ptr)
    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        page_id = tl.load(page_record_ptr + page_slot)
        src_token_idx = page_id * PAGE_SIZE + token_in_page
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        src_token_idx_i64 = src_token_idx.to(tl.int64)
        token_idx_i64 = token_idx.to(tl.int64)
        src_base = src_ptr + src_token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + token_idx_i64 * HIDDEN_SIZE
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        if token_in_page == 0:
            write_idx = (free_list_tail + page_slot) % free_list_capacity
            tl.store(free_list_ptr + write_idx, page_id)
        token_idx += num_blocks
    if pid == 0:
        required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
        new_tail = free_list_tail + required_pages
        tl.store(new_free_list_tail_ptr, new_tail)


# ---------------------------------------------------------------------------
# PagedStashBuffer — pre-allocated paged memory pool
# ---------------------------------------------------------------------------


class PagedStashBuffer:
    """Pre-allocated paged memory pool for stashing activations.

    Uses a flat 2D buffer [total_tokens, hidden_size] with a circular free list
    managed by unwrapped head/tail pointers.

    Args:
        num_tokens: Upper bound on tokens to store.
        hidden_size: Size of the hidden dimension.
        page_size: Number of tokens per page.
        device: Device for the buffer ('cuda' or 'cpu').
        overflow: Shared int64 tensor, set to 1 on OOM.
        dtype: Data type for the buffer.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        device: str | torch.device,
        overflow: torch.Tensor,
        dtype: torch.dtype,
    ):
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.num_pages = (num_tokens + page_size - 1) // page_size
        self.total_tokens = self.num_pages * page_size
        self.dtype = dtype
        self.device = device
        self.overflow = overflow  # shared across buffers

        if str(device) == "cpu":
            self.buffer = torch.empty(
                (self.total_tokens, hidden_size),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            self.buffer = torch.empty(
                (self.total_tokens, hidden_size),
                dtype=dtype,
                device=device,
            )

        # Circular free list with unwrapped head/tail pointers
        self.free_list = torch.arange(
            self.num_pages, dtype=torch.int64, device=device
        )
        self.free_list_head = torch.zeros(1, dtype=torch.int64, device=device)
        self.free_list_tail = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=device
        )
        self.free_list_capacity = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=device
        )

    def __repr__(self):
        return (
            f"PagedStashBuffer(num_pages={self.num_pages}, page_size={self.page_size}, "
            f"hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"
        )

    def reset(self):
        """Reset free list to full capacity. Called per training step."""
        self.free_list.copy_(
            torch.arange(self.num_pages, dtype=torch.int64, device=self.device)
        )
        self.free_list_head.zero_()
        self.free_list_tail.fill_(self.num_pages)


# ---------------------------------------------------------------------------
# PP-aware buffer sizing
# ---------------------------------------------------------------------------


def get_max_in_flight_microbatches(
    parallelism,
    local_batch_size: int,
) -> int:
    """Compute the max number of microbatches with simultaneously stashed
    activations, based on the PP schedule's warmup depth.

    Without PP (pp_degree=1), returns 1 — all stash is from one microbatch.
    With standard 1F1B, the first stage warms up ``min(n_microbatches, pp_degree)``
    microbatches before any backward runs, so that many microbatches have their
    activations simultaneously stashed.
    With interleaved 1F1B, the warmup depth depends on the number of virtual
    stages per rank and the interleaving pattern.

    We use rank=0 (first stage) because it has the highest in-flight count
    across all schedules.

    Args:
        parallelism: ParallelismConfig with PP fields.
        local_batch_size: Local batch size for computing n_microbatches.

    Returns:
        Max number of microbatches with simultaneously stashed activations.
    """
    pp_degree = parallelism.pipeline_parallel_degree
    if pp_degree <= 1:
        return 1

    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    n_microbatches = local_batch_size // microbatch_size
    schedule = parallelism.pipeline_parallel_schedule

    if schedule in ("1F1B", "GPipe"):
        # Standard 1F1B: first stage warms up min(n_microbatches, pp_degree)
        return min(n_microbatches, pp_degree)

    if schedule == "Interleaved1F1B":
        # Interleaved 1F1B formula from torch.distributed.pipelining.schedules
        # _get_warmup_ops(rank=0, n_local_stages, microbatches_per_round,
        #                 pp_group_size, n_microbatches, multiply_factor=2)
        n_local_stages = getattr(
            parallelism, "pipeline_parallel_layers_per_stage", None
        )
        if n_local_stages is None:
            n_local_stages = 1
        number_of_rounds = max(1, n_microbatches // pp_degree)
        microbatches_per_round = n_microbatches // number_of_rounds
        warmups_ops_last_stage = (n_local_stages - 1) * microbatches_per_round
        warmup_ops = warmups_ops_last_stage + 2 * (pp_degree - 1)
        return min(warmup_ops, n_microbatches * n_local_stages)

    # Unknown schedule — conservative upper bound
    logger.warning(
        "Unknown PP schedule '%s' for paged stash sizing; using n_microbatches=%d "
        "as conservative upper bound for max_in_flight",
        schedule,
        n_microbatches,
    )
    return n_microbatches


# ---------------------------------------------------------------------------
# create_paged_buffers — factory
# ---------------------------------------------------------------------------


def create_paged_buffers(model, ac_config, *, max_tokens, max_in_flight=1):
    """Create paged stash buffers for MoE expert activations.

    Scans the model for GroupedExperts modules, counts stash ops per
    (dtype, hidden_size) key, and creates one PagedStashBuffer per key
    sized to the actual number of ops that will use it.

    Per GroupedExperts module, ``_run_experts_grouped_mm`` has 3 ``_grouped_mm`` ops:
      - x @ w1 -> [tokens, hidden_dim]  (key: dtype, hidden_dim)
      - x @ w3 -> [tokens, hidden_dim]  (key: dtype, hidden_dim)
      - h @ w2 -> [tokens, dim]         (key: dtype, dim)
    So 2 ops contribute to (dtype, hidden_dim) and 1 op to (dtype, dim) per module.

    Args:
        model: The transformer model to scan for GroupedExperts.
        ac_config: Activation checkpoint config with paged stash settings.
        max_tokens: Upper bound on tokens routed to experts per step
            (batch_size * seq_len * top_k).
        max_in_flight: Max number of microbatches with simultaneously stashed
            activations. Defaults to 1 (no pipeline parallelism). Use
            ``get_max_in_flight_microbatches()`` to compute from PP config.

    Returns:
        Tuple of (buffers, overflow) where buffers is a dict mapping
        (dtype, hidden_size) to PagedStashBuffer and overflow is the shared
        overflow flag tensor. Returns (None, None) if no GroupedExperts found.
    """
    from collections import defaultdict

    from torchtitan.models.common.moe.moe import GroupedExperts

    device = getattr(ac_config, "paged_stash_buffer_device", "cuda")
    page_size = getattr(ac_config, "paged_stash_page_size", 64)
    buffer_size_factor = getattr(ac_config, "paged_stash_buffer_size_factor", 1.1)

    # Count stash ops per (dtype, hidden_size) key across all GroupedExperts modules.
    ops_per_key: dict[tuple[torch.dtype, int], int] = defaultdict(int)
    num_expert_modules = 0
    for _fqn, mod in model.named_modules():
        if isinstance(mod, GroupedExperts):
            num_expert_modules += 1
            # w1 shape: [num_experts, hidden_dim, dim]
            # w1 and w3 outputs have hidden_dim columns, w2 output has dim columns
            ops_per_key[(mod.w1.dtype, mod.w1.shape[-2])] += 2  # w1 and w3
            ops_per_key[(mod.w1.dtype, mod.w1.shape[-1])] += 1  # w2

    if not ops_per_key:
        logger.warning("No GroupedExperts found; no paged stash buffers created.")
        return None, None

    # Create buffers sized to actual ops per key, scaled by in-flight microbatches.
    overflow = torch.zeros(1, dtype=torch.int64, device=device)
    buffers = {}
    for (dtype, hidden_size), num_ops in ops_per_key.items():
        scaled_max = int(max_tokens * buffer_size_factor * num_ops * max_in_flight)
        buffers[dtype, hidden_size] = PagedStashBuffer(
            scaled_max, hidden_size, page_size, device, overflow, dtype
        )

    logger.info(
        "Created %d paged stash buffers (max_tokens=%d, num_expert_modules=%d, "
        "ops_per_key=%s, max_in_flight=%d, page_size=%d, device=%s)",
        len(buffers),
        max_tokens,
        num_expert_modules,
        dict(ops_per_key),
        max_in_flight,
        page_size,
        device,
    )

    return buffers, overflow
