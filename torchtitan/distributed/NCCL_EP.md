# NCCL EP token dispatch

`NCCLEPTokenDispatcher` supplies fixed-capacity BF16 transport through PyTorch's
TokenSwitch; expert GEMM precision is configured separately. MXFP8 expert
conversion can select FlexGEMM for forward and both backward GEMMs.

Router scores scale the w2 input (as with DeepEP), so combine is a plain sum
and the receive buffer (sized for the worst case, mostly unused) gets no
separate score or mask pass. Its derivative feeds w2's backward directly;
padding rows inside each expert group are zeroed so the weight gradient never
reads uninitialized rows.

`MXFP8NCCLEPTokenDispatcher` keeps the transport's MXFP8 encoding (E4M3 rows,
RCEIL 1x32 E8M0 scales) end to end, so no full-capacity dequantize or
requantize pass runs:

```
dispatch -> MXTensor rows -> w13 (MXFP8) -> act -> * scores_R -> w2 (MXFP8) -> combine
combine' -> MXTensor rows -> w2 dgrad (MXFP8) ...
```

Combine's derivative is then MXFP8 rows, which w2's backward consumes as-is.
This requires MXFP8
grouped w13 and w2 and no `output_postprocess` (checked when `RoutedExperts` is
built), and tracing with GraphTrainer (`make_fx`) or eager: `torch.compile`
raises, because Dynamo gives the MXTensor rows an MXTensor gradient placeholder.

Use an EP degree greater than one, a CUDA EP mesh, and a fixed
`num_max_tokens_per_rank`. BF16 supports `push_pull` and `pull_push`. MXFP8
transport requires `push_pull` on the qualified native version. The identity
permutation bypass requires `NCCL_EP_HT_AG_SCAN_MODE=1` (the AllGather + scan
metadata path, which returns source-ordered rows), plus BF16 `pull_push` or
MXFP8 with `NCCL_EP_HT_RECV_TOPK_ORDER=expert-ascend`; other settings retain
stable sorting.
Native receive ordering can change FP32 accumulation order, so training accuracy
must be qualified separately before promoting that setting.

## Supported training owner

GraphTrainer owns the NCCL EP buffer lifetime and routing capture plan. Its
initialization enters the buffer-owner scope before model buffers are created.
Its warmup observes routing demand; the existing graph `prepare_for_capture`
callback reserves independent routes; capture observes the plan again. Capture
failure releases partial leases. GraphTrainer shutdown drops graphs, releases
routing leases, then closes native buffers before the caller destroys the
process group.

The standard Trainer does not own this lifecycle. Selecting this dispatcher
there raises at buffer initialization, before any native NCCL EP allocation.
No dispatcher-specific cleanup or capture registry is added to the core engine.

Custom callers can own the lifecycle explicitly:

```python
from functools import partial

import torch.distributed as dist

from torchtitan.distributed.accelerator_graph import (
    AcceleratorGraphWrapper,
    accelerator_graph_teardown,
)
from torchtitan.distributed.nccl_ep_capture import (
    NCCLEPBufferOwner,
    NCCLEPRoutingCapturePlan,
)

owner = NCCLEPBufferOwner()
plan = NCCLEPRoutingCapturePlan()
try:
    with owner.initialize():
        dispatcher.init_buffer()
    graph = AcceleratorGraphWrapper(
        partial(plan.run, training_step),
        example_inputs,
        num_warmup_iterations=2,
        prepare_for_capture=plan.prepare_for_capture,
    )
    plan.invoke(graph, *example_inputs)
finally:
    accelerator_graph_teardown()
    plan.release()
    owner.close()
    dist.destroy_process_group()
```

The first two calls run warmup; the third captures. Continue invoking the same
wrapper for replay, keeping shapes and non-tensor inputs fixed. Zero warmup
requires explicitly observed demand before capture. A capture without a plan
raises instead of pinning routes without an owner. Buffer closure is idempotent.
GraphTrainer capture reset releases routing leases while preserving the fixed
input contract; it does not close buffers needed for subsequent capture.

## Saved activations

The main GraphTrainer default save set is unchanged. Experiments can select
`compile.memory_policy="full"` and opt into saved operations through the existing
`compile.full_recompute_save_ops` selectors, for example:

```
*::aten._grouped_mm.default | *::higher_order.flex_gemm | *::aten.sort.stable
```

This explicit policy changes rematerialization and must be qualified for the
selected recipe; it does not reproduce the historical default save-policy patch
merely by selecting the dispatcher.

This integration requires matching PyTorch TokenSwitch compiler, routing,
quantized transport and native lifetime support. The unmodified PyTorch main
interface alone does not supply those APIs. Current-main numerical and GPU
qualification is required; the historical frozen-stack results do not validate
this port or arbitrary NCCL EP releases.
