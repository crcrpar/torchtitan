# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Staged Inductor Configuration for Blackwell GB200

This module provides staged optimization configurations for training on
NVIDIA Blackwell (B100, B200, GB200) GPUs. Use a staged rollout approach
for production training workloads.

Usage:
    # Direct stage setup
    from torchtitan.tools.blackwell_inductor_config import setup_stage1
    setup_stage1()

    # Or via environment variable
    export INDUCTOR_STAGE=2
    from torchtitan.tools.blackwell_inductor_config import setup_from_env
    setup_from_env()

Stages:
    Stage 0 (None): No optimizations, use PyTorch defaults.
    Stage 1 (Conservative): Basic CUTLASS with Blackwell kernels. Start here.
    Stage 2 (Production): Adds Blackwell TMA warp-specialized CUTLASS kernels.
    Stage 3 (Maximum): Exhaustive search. For long training runs only.

Environment Variables:
    INDUCTOR_STAGE: Stage to use (0, 1, 2, or 3). Default: 1
    INDUCTOR_BLACKWELL_DISABLE: Set to "1" to skip all Blackwell optimizations.
    INDUCTOR_PDL: Override PDL setting ("0" or "1").
    INDUCTOR_TMA: Override TMA setting ("0" or "1"). TMA is disabled by default.

Note: Some features from the optimization guide (TMA, persistent TMA matmul,
L1 cache bypass) are disabled by default due to compatibility issues with
the current PyTorch inductor. Use environment overrides to test them.
"""

import os

import torch

from torchtitan.tools.logging import logger


def is_blackwell_gpu() -> bool:
    """Check if the current GPU is a Blackwell architecture (SM100+)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # Blackwell is SM100 (10.0) and SM103 (10.3)
    return major >= 10


def setup_stage1() -> None:
    """
    Stage 1: Conservative baseline - Start here.

    Enables core Blackwell features with minimal risk:
    - Basic CUTLASS integration with Blackwell kernels
    - Default search space (faster compilation)
    - TMA disabled due to potential compatibility issues

    Expected results:
    - Compile time: 2-5 minutes
    - Speedup: ~1.3-1.5x vs eager
    - Stable training with low debugging complexity
    """
    logger.info("Setting up Stage 1: Conservative Blackwell Inductor configuration")

    # NOTE: TMA (use_tensor_descriptor + assume_aligned_inputs) is disabled
    # due to compatibility issues with certain tensor shapes in PyTorch inductor.
    # Enable via INDUCTOR_TMA=1 if you want to test it.
    torch._inductor.config.triton.use_tensor_descriptor = False
    torch._inductor.config.assume_aligned_inputs = False

    # Blackwell epilogue optimization
    torch._inductor.config.triton.enable_epilogue_subtiling = True

    # Basic CUTLASS with Blackwell support
    torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CUTLASS"
    torch._inductor.config.max_autotune_gemm = True
    torch._inductor.config.cutlass.cutlass_epilogue_fusion_enabled = True

    # DEFAULT search space (faster compilation)
    torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"

    # Disable experimental features for stability
    torch._inductor.config.triton.enable_pdl = False
    torch._inductor.config.triton.enable_persistent_tma_matmul = False
    torch._inductor.config.triton.enable_template_tma_store = False
    torch._inductor.config.triton.skip_l1_cache = False

    logger.info("Stage 1 Blackwell configuration applied")


def setup_stage2() -> None:
    """
    Stage 2: Production optimizations.

    Use after Stage 1 is validated stable for 1000+ steps.
    Adds proven optimizations:
    - Blackwell-specific CUTLASS kernels (SM100 TMA warp-specialized)

    Note: Some features from the optimization guide are disabled due to
    compatibility issues:
    - Persistent TMA matmul: Requires TMA which has issues
    - L1 cache bypass: Causes KeyError in buffer tracking

    Expected results:
    - Compile time: 4-6 minutes
    - Additional 5-10% throughput over Stage 1
    - Total speedup: ~1.4-1.6x vs eager
    """
    logger.info("Setting up Stage 2: Production Blackwell Inductor configuration")

    # Start with Stage 1
    setup_stage1()

    # Filter for Blackwell TMA warp-specialized kernels (SM100)
    # This enables the most optimized CUTLASS kernels for Blackwell
    torch._inductor.config.cutlass.cutlass_op_allowlist_regex = (
        "tmawarpspecialized.*sm100"
    )

    # NOTE: The following are disabled due to compatibility issues:
    # - enable_persistent_tma_matmul: Requires TMA which has issues
    # - skip_l1_cache: Causes KeyError in buffer tracking

    logger.info("Stage 2 Blackwell configuration applied")


def setup_stage3() -> None:
    """
    Stage 3: Maximum performance (long training runs only).

    Use only if:
    - Training run > 10,000 steps (amortizes long compile time)
    - Performance is critical
    - You have time to debug if issues arise

    Adds:
    - Exhaustive GEMM search space (more kernel configurations)
    - Exhaustive FlexAttention search

    Note: TMA store operations are disabled due to compatibility issues.

    Expected results:
    - First compile: 15-30 minutes
    - Additional 5-10% over Stage 2
    - Total speedup: ~1.5-1.7x vs eager
    """
    logger.info("Setting up Stage 3: Maximum Blackwell Inductor configuration")
    logger.warning("First compile will take 15-30 minutes!")

    # Start with Stage 2
    setup_stage2()

    # Exhaustive search (adds significant compile time)
    torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"

    # Exhaustive FlexAttention search
    torch._inductor.config.max_autotune_flex_search_space = "EXHAUSTIVE"

    # NOTE: TMA store operations are disabled due to compatibility issues
    # torch._inductor.config.triton.enable_template_tma_store = True

    logger.info("Stage 3 Blackwell configuration applied")


def _apply_env_overrides() -> None:
    """Apply environment variable overrides for individual settings."""
    # PDL override: INDUCTOR_PDL=0 (off) or INDUCTOR_PDL=1 (on)
    pdl_override = os.environ.get("INDUCTOR_PDL")
    if pdl_override is not None:
        enable_pdl = pdl_override == "1"
        torch._inductor.config.triton.enable_pdl = enable_pdl
        logger.info(f"INDUCTOR_PDL override: enable_pdl={enable_pdl}")

    # TMA override: INDUCTOR_TMA=0 (off) or INDUCTOR_TMA=1 (on)
    # TMA is disabled by default due to compatibility issues
    tma_override = os.environ.get("INDUCTOR_TMA")
    if tma_override is not None:
        enable_tma = tma_override == "1"
        torch._inductor.config.triton.use_tensor_descriptor = enable_tma
        torch._inductor.config.assume_aligned_inputs = enable_tma
        logger.info(f"INDUCTOR_TMA override: use_tensor_descriptor={enable_tma}")


def setup_from_env() -> None:
    """
    Setup Blackwell configuration based on environment variables.

    Environment variables:
        INDUCTOR_STAGE: Stage to use (0, 1, 2, or 3). Default: 1
            - 0: No optimizations (use PyTorch defaults)
            - 1: Conservative baseline
            - 2: Production optimizations
            - 3: Maximum performance
        INDUCTOR_BLACKWELL_DISABLE: If set to "1", skip Blackwell optimizations
        INDUCTOR_PDL: Override PDL setting. "0" to disable, "1" to enable.
            By default, PDL is disabled in all stages for stability.
        INDUCTOR_TMA: Override TMA setting. "0" to disable, "1" to enable.
            TMA (Tensor Memory Accelerator) is disabled by default due to
            compatibility issues with certain tensor shapes in PyTorch inductor.

    Example:
        export INDUCTOR_STAGE=2
        python train.py ...

        # Or with TMA enabled (experimental):
        export INDUCTOR_STAGE=1 INDUCTOR_TMA=1
        python train.py ...
    """
    # Check if disabled
    if os.environ.get("INDUCTOR_BLACKWELL_DISABLE", "0") == "1":
        logger.info("Blackwell Inductor optimizations disabled via environment")
        return

    # Check if running on Blackwell
    if not is_blackwell_gpu():
        logger.debug("Not a Blackwell GPU, skipping Blackwell-specific optimizations")
        return

    stage = os.environ.get("INDUCTOR_STAGE", "1")

    if stage == "0":
        logger.info("INDUCTOR_STAGE=0: Skipping Blackwell optimizations")
        return
    elif stage == "1":
        setup_stage1()
    elif stage == "2":
        setup_stage2()
    elif stage == "3":
        setup_stage3()
    else:
        logger.warning(
            f"Unknown INDUCTOR_STAGE: {stage}. Valid values: 0, 1, 2, 3. "
            "Falling back to Stage 1."
        )
        setup_stage1()

    # Apply any environment variable overrides after stage setup
    _apply_env_overrides()
