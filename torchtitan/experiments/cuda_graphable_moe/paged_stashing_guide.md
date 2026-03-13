# Paged Stashing for MoE Expert Activations

A technical guide explaining how paged stashing works, why it exists, and how Megatron-LM and torchtitan implement it differently.

## Background: The Dynamic Shape Problem in MoE + CUDA Graphs

### How MoE Routing Works

In a Mixture-of-Experts (MoE) transformer layer, each token is routed to a subset of experts (typically top-K out of N total experts). The routing decision is made by a learned gating network:

```
Input tokens x [batch * seq_len, dim]
    |
    v
Router: gate(x) -> scores [num_tokens, num_experts]
    |
    v
Top-K selection -> selected_experts [num_tokens, K], top_scores [num_tokens, K]
    |
    v
Reorder tokens by expert assignment -> routed_input [num_tokens * K, dim]
    |
    v
Expert computation (grouped GEMM) -> routed_output [num_tokens * K, dim]
    |
    v
Combine: weighted sum of expert outputs -> output [num_tokens, dim]
```

Each expert is a small FFN (typically SwiGLU: `silu(x @ w1) * (x @ w3)` then `h @ w2`). With `_grouped_mm`, all experts run as a single batched matrix multiply using offset indices, avoiding a per-expert for-loop.

### Expert Parallelism (EP)

With EP, experts are distributed across ranks. Each rank owns `num_experts / ep_size` local experts. Before expert computation, tokens must be sent to the rank that owns their assigned expert (**dispatch**), and results sent back afterward (**combine**). This is an all-to-all communication pattern.

In token-dropless MoE training, the number of tokens received by each expert varies, resulting in a **dynamic shaped tensor**. PyTorch handles this naturally in eager mode -- tensors are allocated lazily when their shape is known at runtime. However, dynamic shaped tensors pose a fundamental challenge for CUDA graphs: there is no CUDA API that allows allocating GPU memory within a CUDA graph with a size determined in stream order.

### The CUDA Graph Dilemma

The most straightforward way to CUDA-graph MoE is to capture using an **oversized buffer** that covers the worst-case token count any EP rank may receive. This works -- the buffer is static, CUDA graph is happy -- but creates significant memory overhead compared to the eager-mode baseline.

The memory problem is twofold:

1. **The compute buffer must be oversized.** CUDA graph requires static shapes, so the dispatch output buffer is pre-sized to worst-case capacity. During compute, the `_grouped_mm` kernels operate on this oversized buffer (padding is skipped via GPU-side offset indices, but the memory is allocated).

2. **Saved activations inherit the oversized shape.** Autograd saves the `_grouped_mm` outputs for the backward pass. These saved tensors have the oversized shape `[max_capacity, hidden_dim]`. Since each step routes different numbers of tokens to different experts, these tensors -- if naively stored -- fragment the memory allocator.

### Paged Stashing: The Key Insight

Paged stashing **decouples** the need for oversized buffers during compute from the need for properly-sized buffers for storing activations:

- **Compute buffers** (oversized, temporary): Used during expert forward/backward kernels. Sized to worst-case capacity. Freed after compute completes.
- **Stash buffers** (right-sized, paged, persistent): Used to store activations for the backward pass. Sized to actual usage. Persist from forward to backward.

The **stash operation** copies the activation from the oversized compute buffer into the paged stash buffer after each expert layer's forward completes. The **restore operation** copies it back during the backward pass. The key memory saving: the stash packs variable-size activations into a contiguous paged buffer, eliminating fragmentation.

For simple scheduling where activation allocation/deallocation follows a first-in-last-out pattern, stash and restore can be done with **bump allocation** (a simple stack pointer). To accommodate complex scheduling (e.g., pipeline parallelism with interleaved fwd/bwd), **paging** provides the flexibility to allocate and free pages in arbitrary order -- hence the name "paged stashing."

### Prerequisites: Reducing Compute Fragmentation

Paged stashing works best when paired with techniques that reduce **compute fragmentation** -- skipping padded data in the oversized static buffers:

- **HybridEP**: Token dispatch kernels that work with pre-sized (worst-case) output buffers without CPU-GPU synchronization.
- **Host-free Grouped GEMM** (`_grouped_mm` with on-device offsets): Expert computation that uses GPU-side offset indices (`torch.cumsum` on device) rather than CPU token counts, avoiding D2H sync.

---

## The Paged Buffer Design

Both Megatron and torchtitan use the same paged buffer design. Each buffer is organized as `[total_tokens, hidden_size]` backed by a circular free list of page IDs:

```
Buffer:     [page_0 | page_1 | page_2 | ... | page_N]   (each page = page_size tokens)
Free list:  [2, 5, 0, 3, 1, 4, ...]                      (available page IDs)
Head ------>                                               (allocate from head)
                                          <---- Tail       (return to tail)
```

Page management is handled by lightweight GPU kernels fused with the stash/restore operations. The Triton `_paged_stash_copy_kernel` allocates pages from the head and copies token data. The `_paged_stash_pop_kernel` copies back and returns pages to the tail. Both operate entirely on GPU -- no CPU involvement.

A `page_record` tensor `[page_id_0, page_id_1, ...]` tracks which pages hold a given activation. This is the compact handle that crosses the fwd->bwd boundary instead of the full activation tensor.

---

## How It Works in Megatron-LM

Megatron's implementation (PR #2690, `vasunvidia/Megatron-LM`, branch `paged_offloading`) operates at the eager/imperative level using PyTorch's `saved_tensors_hooks` API.

### Step 1: The Oversized Buffer is Created at Dispatch

The HybridEP dispatcher pre-sizes the output buffer to worst-case capacity:

```python
# token_dispatcher.py:1012-1020
budget = int(
    routing_map.shape[0]
    * self.config.moe_router_topk
    * self.moe_expert_rank_capacity_factor   # e.g. 1.0 (worst case)
)
self.num_permuted_tokens = budget

# token_dispatcher.py:1052-1063
dispatched_hidden, ... = hybrid_ep_dispatch(
    ...
    num_permuted_tokens=self.num_permuted_tokens,  # oversized
)
# dispatched_hidden shape: [num_permuted_tokens, hidden_dim]
```

This `dispatched_hidden` is the **oversized compute buffer** -- it arrives at `TEGroupedMLP.forward` as `permuted_local_hidden_states`.

### Step 2: `saved_tensors_hooks` Intercept the Save

Megatron wraps each MoE operation in a `PagedStashContext` that installs `saved_tensors_hooks`:

```python
# experts.py:726-737
offload_context = get_paged_stash_context(
    name="expert_fc1",
    max_num_tokens=permuted_local_hidden_states.shape[0],  # oversized dim
    num_tokens_tensor=tokens_per_expert.sum(),              # actual tokens
    avg_num_tokens=int(max_num_tokens // cap_factor),       # heuristic
)
with offload_context:
    fc1_output, bias_parallel = self.linear_fc1(
        permuted_local_hidden_states, tokens_per_expert
    )
```

When `linear_fc1` runs, autograd saves `permuted_local_hidden_states` for the backward pass. The `pack_fn` hook intercepts this:

```python
# paged_stash.py:742-747 (on_save_for_backward)
if (
    self.max_num_tokens is None
    or tensor.dim() == 0
    or tensor.size(0) != self.max_num_tokens   # only intercept oversized tensors
):
    return tensor.detach()   # pass through non-MoE tensors unchanged
```

Only tensors whose first dimension equals `max_num_tokens` (the oversized budget) are intercepted. All other tensors pass through as plain detached tensors. This heuristic is effective but can theoretically produce false positives.

The intercepted tensor is wrapped in a `PagedTensor`:

```python
# paged_stash.py:810-824
paged_tensor = PagedTensor(
    tensor,                         # the oversized activation
    num_tokens_tensor=...,          # actual token count (GPU scalar)
    max_tokens=self.max_num_tokens, # the oversized dim
    page_size=self.page_size,
)
```

### Step 3: The Stash Copy Happens Asynchronously

After expert compute completes, `paged_stash_group_commit` launches the stash on a separate CUDA stream:

```python
# paged_stash.py:605-625 (stash_paged_tensors)
def stash_paged_tensors(self, pp_schedule_layer):
    current_stream = torch.cuda.current_stream()
    self.pack_stream.wait_stream(current_stream)     # pack_stream waits for compute

    with torch.cuda.stream(self.pack_stream):        # async on pack_stream
        while len(self.paged_tensors_to_stash) > 0:
            paged_tensor = self.paged_tensors_to_stash.pop(0)
            stash_buffer = self.stash_buffers[paged_tensor.dtype][paged_tensor.hidden_size]
            paged_tensor.offload_to_stash(stash_buffer)    # Triton kernel
            self.paged_tensors_stash_in_progress.append(paged_tensor)
```

Inside `offload_to_stash`, after launching the Triton copy kernel:

```python
# paged_stash.py:367-369 (offload_to_stash)
self._original_tensor = self._tensor   # keep reference (copy still in-flight)
self._tensor = None                    # clear autograd's reference
```

At this point, `_original_tensor` holds the oversized buffer alive (the async copy needs it), but `_tensor` is cleared so autograd doesn't reference it.

### Step 4: The Oversized Buffer is Freed

This is the critical question: **when does the oversized buffer actually get freed?**

It happens in the **next layer's** `paged_stash_group_start`, which calls `wait_for_stash_to_complete`:

```python
# paged_stash.py:632-645 (wait_for_stash_to_complete)
def wait_for_stash_to_complete(self):
    current_stream = torch.cuda.current_stream()
    if self._pack_stream_status == 'stashing':
        current_stream.wait_stream(self.pack_stream)    # sync: Triton copy done
        self._pack_stream_status = 'idle'

        # FREE the oversized buffers from the previous layer:
        while len(self.paged_tensors_stash_in_progress) > 0:
            paged_tensor = self.paged_tensors_stash_in_progress.pop(0)
            paged_tensor._original_tensor = None   # <-- THIS is the actual free
```

Setting `_original_tensor = None` drops the last Python reference to the oversized tensor. PyTorch's caching allocator returns its memory to the pool. The `wait_stream` at line 636 guarantees the Triton copy has completed first.

**Why wait in the next layer's `group_start` and not immediately after `group_commit`?** To reduce peak memory. The next expert layer will allocate its own oversized compute buffer. If we freed the previous buffer at `group_commit` time, we'd need to synchronize immediately (blocking the main stream). By deferring to the next `group_start`, the stash copy runs concurrently with other non-expert compute.

### The Complete Timeline

```
Layer N forward:
  group_start  -> wait_for_stash_to_complete()
                    -> sync pack_stream (previous layer's stash is done)
                    -> _original_tensor = None  (free layer N-1's oversized buffer)
  fc1 inside paged_stash_context
                    -> on_save_for_backward intercepts oversized input
                    -> wraps in PagedTensor, queues in paged_tensors_to_stash
  fc2
  group_commit -> stash_paged_tensors() on pack_stream (async)
                    -> Triton copy: oversized buffer -> paged stash buffer
                    -> _original_tensor = _tensor; _tensor = None

Layer N+1 forward:
  group_start  -> wait_for_stash_to_complete()
                    -> sync pack_stream (layer N's stash is done)
                    -> _original_tensor = None  (free layer N's oversized buffer)
```

### The 3-Phase State Machine

**Phase 1 -- `capture` (iteration 1):**
Runs a real forward+backward pass. The `pack_fn` tracks peak concurrent stash usage via a running counter (increment on save, decrement on backward get). This captures the high-water mark per `(dtype, hidden_size)`.

**Phase 2 -- `captured` (iteration 2):**
Buffers are allocated from the high-water mark: `num_tokens = max_tokens[key] * stash_buffer_size_factor`. The capture step doubles as the CUDA graph warmup (needed anyway).

**Phase 3 -- steady state (iteration 3+):**
CUDA graphs are active. Each step resets buffer free lists, replays the stash/restore pattern recorded during capture.

---

## How It Works in torchtitan

torchtitan's implementation (`experiments/cuda_graphable_moe`) operates at the **FX graph level** using AOT compilation with graph passes.

### Step 1: The Oversized Buffer is Created at Dispatch (Same Mechanism)

HybridEP pre-sizes the dispatch output the same way:

```python
# hybridep.py:108-120
def _num_permuted_tokens_for_non_blocking(
    num_tokens, ep_size, num_local_experts, top_k, moe_expert_capacity_factor,
) -> int:
    n = int(num_tokens * ep_size * min(num_local_experts, top_k) * moe_expert_capacity_factor)
    return maybe_align_num_tokens_for_mxfp8(n)
```

During AOT tracing, the fake tensor dispatch produces this oversized shape:

```python
# hybridep.py:244-245 (fake impl for tracing)
out_tokens = _num_permuted_tokens_for_non_blocking(...)
hidden = x.new_empty(out_tokens, x.shape[1])   # [oversized, hidden_dim]
```

So the joint graph's `_grouped_mm` nodes are traced at their oversized shapes.

### Step 2: FX Annotation Marks Target Ops (Instead of `saved_tensors_hooks`)

Instead of runtime hooks, torchtitan annotates at trace time:

```python
# deepseek_v3/parallelize.py:80-86
import torchtitan.models.common.moe.moe as moe_module
moe_module._run_experts_grouped_mm = annotate_fn({"paged_stash": True})(
    moe_module._run_experts_grouped_mm
)
```

After tracing, every FX node from `_run_experts_grouped_mm` carries `node.meta["custom"]["paged_stash"] = True`. The joint graph pass uses this:

```python
# paged_stash_graph_pass.py (apply_paged_sac_pass)
for node in gm.graph.nodes:
    if node.meta.get("custom", {}).get("paged_stash", False):
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
```

This tells the min-cut partitioner: these `_grouped_mm` outputs become **fwd->bwd saved tensors** (edges from the forward graph to the backward graph).

### Step 3: What Happens to the Oversized Buffer?

In the AOT path, the lifecycle is different from Megatron's eager path:

1. **The dispatch output** (`hidden` from HybridEP) is the oversized buffer `[max_capacity, dim]`. It flows into `_run_experts_grouped_mm` as input `x`.

2. **`_grouped_mm` outputs** (3 per module) have shape `[max_capacity, hidden_dim]` or `[max_capacity, dim]`. These are the tensors marked `MUST_SAVE` by `apply_paged_sac_pass`.

3. **The min-cut partitioner** places these in the fwd graph output / bwd graph input. They cross the fwd->bwd boundary as saved tensors.

4. **During CUDA graph capture**, these saved tensors are allocated at their oversized shape inside the CUDA graph memory pool. Their addresses are baked into the graph.

5. **During CUDA graph replay**, the same addresses are reused each step. The `_grouped_mm` kernels write to the oversized buffer, but only process `actual_tokens` rows (via GPU-side offsets from `cumsum`). The padded rows contain garbage but are never read.

The oversized compute buffer (dispatch output `hidden`) is NOT saved for backward -- `hybridep::dispatch`'s `save_for_backward` only saves `topk_idx` (the index tensor), not the oversized `hidden`:

```python
# hybridep.py:311-317
def _dispatch_setup_context(ctx, inputs, output):
    x, topk_idx, _, _, _, _, dispatch_handle = inputs
    ctx.save_for_backward(topk_idx)   # only indices, not the oversized hidden
```

So the dispatch buffer is freed after expert compute uses it. The `_grouped_mm` outputs are what persist as saved tensors.

### Step 4: Buffer Allocation is Static (No Capture Iteration)

Unlike Megatron's runtime capture, torchtitan sizes buffers from model structure:

```python
# paged_stash_ac.py:create_paged_buffers
ops_per_key: dict[tuple[torch.dtype, int], int] = defaultdict(int)
for _fqn, mod in model.named_modules():
    if isinstance(mod, GroupedExperts):
        ops_per_key[(mod.w1.dtype, mod.w1.shape[-2])] += 2  # w1 + w3
        ops_per_key[(mod.w1.dtype, mod.w1.shape[-1])] += 1  # w2

scaled_max = int(max_tokens * buffer_size_factor * num_ops * max_in_flight)
```

The `max_tokens` matches the HybridEP dispatch buffer size (same capacity factor formula). The `max_in_flight` factor accounts for pipeline parallelism.

### Step 5: CUDA Graph Captures at Oversized Shapes

The `CUDAGraphWrapper` captures the compiled fwd/bwd graphs:

```python
# cudagraph.py:109-140
if self.cudagraph is None:
    self.args = args          # static input buffer addresses
    self.cudagraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.cudagraph, pool=self.graph_pool, stream=self.stream):
        self.output = self.runnable(*args)   # all tensor allocations captured here
```

During replay, `copy_non_static_inputs` copies new token data into the static (oversized) input buffers. The paged stash buffer addresses are module attributes -- they never change across replays.

### Accounting: Where Ideas Map Between Implementations

| Concept | Megatron | torchtitan |
|---|---|---|
| **Oversized compute buffer** | `dispatched_hidden` from `hybrid_ep_dispatch` `[max_capacity, dim]` | Same: `hidden` from `dispatch_tokens` at traced oversized shape |
| **Which tensors are stashed** | Heuristic: `tensor.size(0) == max_num_tokens` | Explicit: `annotate_fn({"paged_stash": True})` on `_run_experts_grouped_mm` |
| **Stash interception point** | `saved_tensors_hooks` pack/unpack at runtime | FX annotation + `apply_paged_sac_pass` at compile time |
| **When oversized buffer is freed** | Explicitly: `_original_tensor = None` in next layer's `group_start` after sync | Implicitly: AOT graph manages tensor lifetimes; dispatch buffer not saved for backward |
| **Stash/restore overlap** | Explicit: `pack_stream` / `unpack_stream` with Pre/Post-scheduler autograd Functions | Implicit: CUDA graph captures the execution order; overlap baked into the graph |
| **Buffer sizing** | Runtime high-water mark from capture iteration | Static: `ops_per_key * max_tokens * buffer_size_factor * max_in_flight` |
| **CUDA graph integration** | 3-phase state machine (begin -> capture -> captured) | Separate `CUDAGraphWrapper` (warmup -> capture -> replay) |
| **Page management** | Triton kernels with circular free list | Same Triton kernels, same free list design |
| **PP support** | Runtime capture tracks interleaved fwd/bwd via increment/decrement counters | Static `max_in_flight` from PP schedule warmup formulas |

---

## Pipeline Parallelism Considerations

### Why PP changes buffer sizing

Without PP, all forward activations are saved before any backward runs. Peak concurrent stash = total stash. The static formula `ops_per_key * max_tokens` is exact.

With PP (especially interleaved 1F1B), forward and backward microbatches overlap:

```
Standard 1F1B (4 stages, 8 microbatches):
  F1 F2 F3 F4 | F5 B1 F6 B2 F7 B3 F8 B4 | B5 B6 B7 B8
  ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
  warmup (4)     steady state (1F1B)         cooldown
```

During warmup, the first stage accumulates 4 microbatches of stashed activations before any backward frees pages. At steady state, one backward frees while one forward creates -- the peak is the warmup depth.

### torchtitan: Static schedule analysis

`get_max_in_flight_microbatches` computes the worst-case warmup depth from PP schedule parameters:

| Schedule | `max_in_flight` (worst-case, rank 0) |
|---|---|
| No PP (`pp=1`) | 1 |
| 1F1B / GPipe | `min(n_microbatches, pp_degree)` |
| Interleaved 1F1B | `(n_local_stages - 1) * microbatches_per_round + 2 * (pp_degree - 1)` |

Formulas from PyTorch's `torch.distributed.pipelining.schedules` (`Schedule1F1B._step_microbatches` and `_get_warmup_ops`).

### Megatron: Runtime capture

Megatron's `PagedStashManager` captures the actual interleaved execution order during iteration 1. The `on_save_for_backward` increments a counter, `on_get_saved_tensor` decrements it. The high-water mark across the full fwd+bwd pass (including PP interleaving) is the exact peak. This is more general but costs one training iteration (which doubles as CUDA graph warmup).

### When they diverge

For standard schedules (1F1B, GPipe, Interleaved 1F1B), the static formula gives the same answer as runtime capture. They diverge for exotic schedules, VP stages with heterogeneous layer counts, or dynamic microbatch sizing.

### Existing PyTorch infrastructure

Several tools could extend PP-aware analysis:

- **`MemoryTracker`** (`torch._inductor.fx_passes.memory_estimator`): Tracks live memory during node scheduling. Used by inductor's overlap scheduler to enforce memory budgets.
- **`build_memory_profile`** (same module): Simulates FX graph execution, returns memory time-series. Infrastructure for future memory-aware reordering.
- **`get_schedule_ops`** (`torch.distributed.pipelining._schedule_visualizer`): Generates full action lists for any PP schedule. Could count in-flight microbatches at each timestep.

None are PP-aware today, but the building blocks exist.

---

## Comparison Summary

### Architecture

| Dimension | Megatron | torchtitan |
|---|---|---|
| **Abstraction level** | Eager -- hooks intercept autograd at runtime | Graph-level -- FX passes annotate before compilation |
| **Stash granularity** | Per-op: fc1, activation, fc2 independently configurable | Per-function: all ops inside `_run_experts_grouped_mm` |
| **Buffer allocation** | Deferred to iteration 2 (runtime profiled) | Pre-allocated (statically computed from model + schedule) |
| **Stash/restore overlap** | Explicit async streams + Pre/Post-scheduler | Implicit: baked into CUDA graph execution order |

### Open Questions

- **False positives in tensor selection**: Megatron's `size(0) == max_num_tokens` heuristic can match unrelated tensors. torchtitan avoids this with explicit FX annotation.
- **Capture step requirement**: Megatron needs one profiling iteration. torchtitan's static analysis avoids this but may not handle exotic PP schedules.
- **Composition with AC**: Both compose -- torchtitan via `apply_sac` + `apply_paged_sac` joint passes, Megatron via `stash_modules` config selecting which ops to stash vs. recompute.
