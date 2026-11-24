# Project Cleanup and Extensibility Contract

## Status and scope

This document is an audit and implementation contract for the Episodic-Memory-in-RL
repository. It describes the behavior that cleanup work must preserve and the
interfaces that future work, including synthetic gradients, must use. It does not
authorize or specify an implementation change by itself.

The repository has pre-existing modified and untracked files. An implementation pass
must start from and preserve those edits. The named worktree changes include
`configs/TrainConfig.py`, `datahandling/datasets/EpisodeDataset.py`,
`experiments/tmaze.py`, `models/TBD/TBD_Block.py`, `train.py`,
`trainers/BaseTrainer.py`, `reproducibility.py`, `pyproject.toml`, and `uv.lock`.

## 1. Current behavior

### 1.1 Experiment entry points

The supported batch entry point is:

1. `dispatch_runs.py` parses an experiment module and seed count.
2. Without `--no-slurm`, it submits `run_experiment.sh` as a Slurm array.
3. The array script activates the configured environment and runs
   `python3 run_experiment.py "$1"`.
4. `run_experiment.py` imports the named experiment module and calls `run(index)`.
5. The seed index comes from `--index` when supplied, otherwise
   `SLURM_ARRAY_TASK_ID`.

There is no file named `run_experimenty.py` in the repository. This is treated as
a typo in the request unless one is created explicitly. `run_experiment.py` is the
Python entry point and `run_experiment.sh` is the Slurm wrapper.

The no-Slurm branch of `dispatch_runs.py` is currently broken: it imports
`run_experiment` as a function, but `run_experiment.py` currently only exposes a
module guard and a CLI `run` call.

### 1.2 Configuration and construction

Dataclasses in `configs/` and model-specific config modules are used as factories.

- `RunConfig` aggregates model, environment, dataset, and training configuration.
- `ModelConfig.get_model()` constructs a complete model.
- Layout registration in `models/DTi/Config.py` dynamically assembles blocks.
- `DTi/layouts.py` selects ordinary Transformer blocks or Titans-style memory blocks.
- `train.py` seeds work, constructs the model, optionally compiles it, and starts the
  trainer.
- `BaseTrainer.train()` owns the data loader, evaluation environment, optimizer,
  checkpointing, WandB setup, and batch loop.

This split is useful, but construction and execution responsibilities are currently
mixed. Trainer subclasses contain only `iter()` implementations, while base trainer
code owns most of the training lifecycle.

### 1.3 Training

There are two trainer modes:

- `SubsequenceTrainer`: resets state once, consumes one full batch, performs one
  forward pass and one loss/backward/step.
- `EpisodeTrainer`: resets state once, processes an episode in `context_len` chunks,
  optionally detaching state every `detach_every` chunks, concatenates action
  predictions, and performs one loss/backward/step.

Both use `AdamW` over `model.parameters()`, gradient clipping, and one-step updates.

The episode trainer currently snapshots state for evaluation by
`model.get_states()` and restores it with `model.set_states()`. It does not reset
evaluation memory, and the cacheful/sampling state is also part of the ambient model
state.

### 1.4 Stateful model state

`Stateful` defines reset, partial reset, get, set, and detach operations.
`BaseModel` discovers implementations through the module tree and exposes aggregate
operations.

Fast-memory implementations own mutable tensors:

- `TTT_Memory`: per-batch memory TensorDict.
- `TitansMemory`: per-batch memory and momentum TensorDicts.
- `FwPKM_Memory`: per-batch key and value tensors.
- Their states are reset at batch start, detached between episode chunks, restored
  around evaluation, and copied into state snapshots.

Titans writes use the read-first/write-after sequence and a meta-gradient update to
train the fast weights at test-time. Memory modules are therefore not ordinary stateless PyTorch
parameters: they combine static learned parameters with mutable runtime fast states.

### 1.5 Cacheful inference state

`Cacheful` modules maintain an attention/write cache for chunked inference.
`BaseModel` discovers and resets/wipes/enables these caches through the module tree.

The chunked sampler enables caches, executes one token or chunk at a time, performs a
cache reset at chunk boundaries, and restores the model to training mode after
sampling. The sliding-window sampler resets caches for every step and does not use
chunk persistence.

Cache state is separate from `Stateful` fast-memory state, but both are traversal-
discovered state and are currently implicit in `model.train()`/`model.eval()` behavior.

### 1.6 Current Titans and synthetic-gradient placeholder

`SG_Block.py` mirrors `MAL_Block.py` and contains an unused `grads_pred` network:

- Input size is the total parameter count of `mem`.
- Output size is the same parameter count.
- It is not called or used in a loss.
- It is not registered in a layout.
- No synthetic-gradient target or gradient path exists yet.

This is a placeholder for future synthetic-gradient work, not a functioning feature.

## 2. Dependency injection versus state ownership

### Decision

Use a **hybrid contract**:

1. Inject static dependencies.
2. Let memory modules own fast-weight **data**, but expose that state explicitly.
3. Make a run/training episode hold an explicit runtime-state object.
4. Do not let the trainer mutate arbitrary module state through module-tree discovery.

Static dependencies should be injected as factories or interfaces:

- memory implementation and memory dimensions,
- memory learning/update policy,
- loss and target construction,
- optimizer,
- gradient-clipping policy,
- sampler,
- checkpoint/store,
- metrics/logger,
- synthetic-gradient predictor.

Fast weights and momentum are better owned by the memory module. They are the natural
implementation state of the memory algorithm, and centralizing all TensorDict internals
in a generic trainer would leak algorithm details. However, they must be represented
through a serializable, explicit runtime-state API rather than hidden attributes.

The generic model may compose modules and coordinate state, but it should not discover
statefulness solely through `isinstance(module, Stateful)`. A module should advertise
its runtime state; a model/run context should request, clone, replace, and detach it.

Recommended conceptual types:

- `StaticConfig`: dataclasses and immutable construction settings.
- `RuntimeState`: per-stream memory, momentum, cache, and step counters.
- `ForwardContext`: immutable request metadata such as stream index, device, batch,
  training mode, chunk boundary, and whether to retain/return state.
- `TrainingBatch`: model inputs, targets, masks, sequence lengths, and optional
  gradient targets.

For current code, preserve `Stateful` and `Cacheful` compatibility during migration,
but add deprecation-targeted aggregate methods only after call sites use an explicit
runtime-state context.

## 3. Synthetic-gradient contract

Synthetic-gradient support is an optional `TrainingExtension`, not a dependency of
`TrainConfig`, `BaseTrainer`, or `OptimizationPolicy`. The base pipeline owns only
primary-task forward/loss/backward/step behavior. A synthetic-gradient extension
attaches explicitly to a trainer, receives before/after-batch hooks, and owns its
predictor state and checkpoint state. Synthetic-gradient implementations must not
require `OptimizationPolicy` to recognize a `"synthetic"` loss key.

Synthetic gradients are a future opt-in training objective. They predict gradients
that would be produced beyond a stopped or truncated BPTT boundary.

The implementation must support TD(0)-style bootstrapping:

`predicted_gradient_t = predictor(..., true_or_bootstrapped_gradient_t+1)`

The exact synthetic-gradient target depends on the future computation graph, but the
system must not hard-code the target into a trainer or block.

### 3.1 Required modular capabilities

A training setup for synthetic gradients must provide:

1. A forward execution API that can return both model outputs and per-step state.
2. A graph boundary API that can stop, detach, checkpoint, or resume gradient flow.
3. At least two unrolled forward steps for a write operation:
   - Step 1 writes or updates fast weights.
   - Step 2 reads or otherwise exposes the effect of that write to produce the
     objective-dependent gradient signal.
4. A target adapter that records or predicts actual/bootstrapped gradients outside
   the allowed BPTT cutoff.
5. Separate optimizer or parameter groups for:
   - base model parameters,
   - fast-memory update parameters,
   - synthetic-gradient predictor parameters.
6. A loss decomposition that keeps the primary task loss and synthetic-gradient loss
   independently measurable.

No synthetic-gradient implementation may rely on mutating `mem.parameters()` or
flattening fast weights in a block constructor. A predictor should receive an explicit
gradient/state tensor descriptor and expose a shape contract.

### 3.2 State and BPTT boundaries

The unrolled sequence must distinguish:

- in-graph state (gradients may flow),
- retained state (gradients stop after this checkpoint),
- bootstrapped state (future gradient is supplied by the predictor),
- cache state (sampling acceleration only and never a training parameter update).

Episode chunking should be implemented as a sequence of explicit `StepSpec` values
rather than as hard-coded `for i in range(...)` loops.

Minimum `StepSpec` fields:

- input slice,
- state snapshot to restore,
- whether to detach state before forward,
- whether to return activations/gradients,
- whether state may be modified,
- optional synthetic target/horizon.

### 3.3 Predictor contract

The predictor is expected to support the following conceptual interface:

```python
forward(context: Tensor, state_descriptor: Tensor) -> Tensor
loss(predicted: Tensor, target: Tensor) -> Tensor
```

The predictor must not infer a whole memory module's parameter shape from
`sum(p.numel() ...)` inside a block constructor. It must receive and validate an
explicit `GradientTensorSpec` or state-flat/structured format.

For Titans, the predictor input may be a structured representation of:

- base read/write inputs,
- memory parameters,
- momentum,
- target horizon,
- the next write/read context,

but no implementation may assume the memory state is a single fixed flat vector.
The predictor output shape must be separately declared and checked against target
gradient shapes.

### 3.4 Synthetic-gradient update sequence

The baseline sequence is:

1. Reset/restore runtime state.
2. Run at least two steps at a BPTT boundary.
3. Compute the visible objective gradient for the first retained window.
4. Run or predict the gradient beyond the cutoff.
5. Compute the synthetic predictor loss:
   - target is an actual or detached future gradient, or
   - target is the predictor's TD bootstrapped estimate.
6. Backpropagate to each distinct parameter set according to the configured objective.
7. Clip and update the relevant optimizer(s).
8. Record primary loss, predictor loss, finite checks, and horizon coverage.

An implementation must expose a switch to disable synthetic gradients and reproduce
the current behavior exactly enough for baseline comparisons.

## 4. Batch and supercomputer contract

The batch system must have stable and independently callable interfaces.

### 4.1 Required CLI

Add a Python API:

```python
run_experiment(experiment: str, index: int) -> None
```

and preserve the CLI:

```bash
python3 run_experiment.py EXPERIMENT_MODULE [--index INDEX]
```

The CLI must resolve an index in this order:

1. `--index`, if present.
2. `SLURM_ARRAY_TASK_ID`, if present.
3. A clear error if neither is available.

The no-Slurm dispatcher must call the Python API directly and must not require
`SLURM_ARRAY_TASK_ID`.

`run_experiment.sh` must:

- verify required environment variables and arguments,
- create `logs/` before Slurm opens output/error files,
- pass both the experiment module and array index explicitly,
- fail with a non-zero exit code on experiment failure,
- avoid silently succeeding after a missing module or missing seed.

The script may remain institution-specific, but the settings should be overridable
through environment variables or CLI arguments instead of hard-coded credential,
partition, GPU, memory, or module assumptions when practical.

### 4.2 Seed and reproducibility

Each array element must receive a unique, explicit seed derived from `index`.
The seed must be written to checkpoint and WandB metadata. Resume and multi-worker
reproducibility must use the same seed policy.

### 4.3 Checkpoints

A checkpoint must include:

- model state,
- every optimizer state,
- batch number,
- run/experiment seed,
- experiment/module configuration,
- WandB run id,
- runtime-state format version,
- optional synthetic-gradient predictor state.

Resume must reject incompatible config/state formats with a descriptive error rather
than silently restoring mismatched shapes.

## 5. Coding smells and cleanup priorities

### P0: correctness and reproducibility

1. Fix `dispatch_runs.py` importing a nonexistent `run_experiment` function.
2. Correct `envionment` to `environment` in `RunConfig` and `train.py`.
3. Do not call the module-traversal state APIs for state that should belong to a
   newly explicit runtime context; preserve compatibility while migrating.
4. Replace ambiguous bare asserts as runtime validation in public APIs where invalid
   state is possible. Assertions should not be the only cache/state initialization
   contract.
5. Add tests for:
   - no-Slurm dispatch,
   - Slurm index fallback,
   - missing index/module errors,
   - reset/partial reset/get/set/detach state,
   - cache reset/wipe/enable,
   - chunk and episode boundaries,
   - synthetic loss shape and no-synthetic baseline.

### P1: design and separation

1. Split `BaseTrainer.train()` lifecycle setup from the batch/optimization algorithm.
   A trainer should compose data, state, forward, loss, optimizer, logging, and
   checkpoint strategies rather than perform all setup inline.
2. Introduce a small training-loop protocol:
   - `make_runtime_state`
   - `next_batch`
   - `forward`
   - `loss`
   - `backward_and_step`
   - `checkpoint/restore`
   - `evaluate`
3. Separate primary task objective, synthetic-gradient objective, and BPTT boundary
   policy.
4. Replace layout `inspect.signature` reflection with explicit layout dataclasses or
   typed layout registration. The current signature matching is convenient but fragile
   and produces opaque missing-field errors.
5. The public name for memory update frequency is now `memory_update_interval`: it
   is the number of residual-stream tokens processed between fast-memory writes.
   `mini_batch_size` remains as a backward-compatible constructor/config name for
   existing call sites, but it does not mean a training batch size.
6. Remove or replace wildcardish mutable defaults:
   - `mem_module` is a factory object used as a default argument.
   - mutable runtime objects are represented as optional fields initialized outside
     constructors and validated later.

### P1: state and cache correctness

1. Define whether `reset_states(batch_size)` occurs before or after the dataloader
   batch is moved to device.
2. Define cache behavior for ragged/partial chunks.
3. Verify state keys and cache shapes in `set_states()` and runtime restoration.
4. Evaluate whether cache state should be restored/snapshotted around evaluation.
5. Ensure sampling always unwinds cache state even on exceptions or `break` paths.
   Prefer context-managed sampling state.
6. Treat cache state and fast-memory state as separate runtime concerns; do not infer
   either from `training`/`eval` alone.

### P2: API and maintainability

1. Add module and public API docstrings for training contracts.
2. Avoid comments that describe uncertainty as implementation decisions, including
   "worth further investigation" or "I am not sure" in block constructors.
3. Replace duplicate `MAL_Block` and `SG_Block` structure with shared read/write/cache
   machinery and an optional objective/predictor component.
4. Remove unused imports and variables as cleanup proceeds.
5. Pin and document the package-manager workflow used to create `uv.lock`.
6. Add a repository test suite and a minimal CPU-smoke test so CUDA-only code does
   not prevent baseline validation.

## 6. Acceptance criteria

Cleanup is complete when:

- Existing DT, Titans/MAL, TBD, episode, and subsequence behavior is covered by tests.
- Batch dispatch works with and without Slurm and reports invalid arguments clearly.
- State and cache lifecycle APIs are explicit and serializable.
- Synthetic gradients can be enabled or disabled without editing block internals.
- Synthetic-gradient training supports a configurable horizon and at least two
  unrolled steps.
- Predictor targets and outputs have validated, versioned shapes.
- Primary loss, synthetic loss, clipping, and optimizer updates are independently
  visible in logs and metrics.
- A checkpoint can restore model, optimizers, runtime state, and synthetic predictor
  state.
- The project can run a CPU smoke test in an environment without a GPU.
- No production behavior depends on module traversal alone for required state.

## 7. Recommended implementation order

1. Add tests and repair the batch entry point and configuration typo.
2. Extract immutable forward/training batch data and explicit runtime-state context.
3. Preserve existing `Stateful`/`Cacheful` implementations behind adapters.
4. Extract the optimization step and loss decomposition from trainer subclasses.
5. Add synthetic-gradient interfaces as opt-in components with a disabled baseline.
6. Implement the Titans predictor and TD-style future-gradient target.
7. Add checkpoints and Slurm metadata for the new stateful components.
8. Remove traversal-based state APIs only after all call sites have migrated.
