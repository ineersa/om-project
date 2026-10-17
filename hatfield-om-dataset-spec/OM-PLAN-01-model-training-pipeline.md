# OM-PLAN-01: Observational Memory model training pipeline

**Status:** proposed project plan  
**Audience:** the coding agent implementing the pipeline  
**Companion specification:** `OM-DATA-01-hatfield-memory-dataset-spec.md` (named `OM-DATA-01.md` inside the specification package)

**Follow-up:** use [OM-PLAN-02: Judge-guided memory writer training](OM-PLAN-02-judge-guided-training.md) for the next-stage work order and candidate-selection policy. Keep good existing candidates rather than requiring new teacher outputs for every case. Use OM-DATA-01's current implementation boundaries rather than the PHP runtime bridge proposed below.

## 1. What we are building

Build an **offline, reproducible pipeline that turns historical Hatfield and Pi sessions into a validated training dataset, fine-tunes a small Observational Memory model, and measures whether it is a useful replacement for the current Qwen deployment.**

This is not a project to train the main coding assistant. We are training the specialist that decides what the assistant should remember as conversation history leaves its context.

The initial student candidate is `LiquidAI/LFM2.5-2.6B`. The comparison baseline is the user's deployed Qwen3.6-35B-A3B configuration, including its actual checkpoint, quantization, prompts, and serving settings. Both are configuration choices, not hardcoded assumptions.

Although we have called this “pretraining an OM model,” **the first experiment is supervised fine-tuning (SFT), starting with LoRA, using validated examples from a stronger teacher.** Training from scratch, continued pretraining on raw transcripts, and reinforcement learning are not part of the first version.

The deliverable is more than a checkpoint: it is a repeatable process for producing datasets, training candidates, comparing them, and deciding whether to deploy them. A smaller model beating Qwen is a hypothesis to test, not a promised outcome.

## 2. The behavior we want to teach

The project has three related tasks:

| Task | Model input | Model responsibility |
|---|---|---|
| **Observer** | Visible prior memory and a new source-addressed conversation chunk | Record new, useful observations with accurate timestamps, relevance, and evidence references. |
| **Reflector** | Current observations and reflections | Add durable conclusions without inventing facts or claiming to preserve details that were lost. |
| **Dropper** | Active observations, reflections, and removal limits | Propose observations safe to remove from active memory; the runtime still validates, ranks, and caps removals. |

**Start with Observer.** Keep Reflector and Dropper fixed during the initial replacement experiment, while preserving their data and contracts for later work. A shared model for all three tasks is a later experiment, not a prerequisite.

The training target is the real memory-tool interaction, including valid no-action decisions and correct stopping—not a prose summary that an adapter later tries to reinterpret.

These task boundaries follow the companion specification, particularly §§1 and 5–6. Its schemas and runtime contracts remain authoritative.

## 3. End-to-end pipeline

```text
Historical Hatfield sessions + OM database + old Pi sessions
    -> read-only snapshots and corpus inventory
    -> privacy filtering, version adapters, lineage mapping, frozen splits
    -> causal memory-agent invocation construction
    -> source-first teacher generation and comparison with Qwen outputs
    -> deterministic validation, independent judging, human review
    -> versioned accepted dataset and model-specific training export
    -> Observer SFT/LoRA experiments
    -> fixed-input evaluation, whole-history memory replay, downstream tests
    -> deployment decision, shadow evaluation, controlled rollout
```

Each stage produces inspectable artifacts. Stages must be independently runnable and resumable. A failure or ambiguous record goes into an explicit error/review queue rather than silently becoming training data.

### Stage A — Acquire and understand the source corpus

Import complete Hatfield journals, consistent OM database snapshots, relevant settings, and available memory-agent traces. Import Pi session files with their native ancestry and memory-ledger entries.

Preserve originals and hashes. Never modify or migrate live session data. A database snapshot and journal export are not automatically one atomic snapshot; record boundaries and check that referenced events exist.

Classify what is actually available: exact captured requests, faithfully reconstructable requests, raw history suitable for fresh replay, or summary-only records. Unknown Pi versions and missing evidence must be visible in the inventory.

**Deliverable:** an inventory and immutable private archive showing usable sessions, independent session families, reconstruction limitations, and input-length distributions.

### Stage B — Normalize, protect, and split the data

Create version-specific Hatfield and Pi adapters. Preserve native IDs while mapping them to a common representation and the target Hatfield tool contract. Do not assume all old Pi sessions use the currently inspected format.

Remove secrets before any remote labeling call. Preserve consistent redacted identities within a session family and keep the restricted mapping out of model exports.

Group related sessions, forks, continuations, copied histories, and near-duplicates. Freeze train/validation/test assignments **by independent session family or work episode before labeling or augmentation**. Derived windows and synthetic variants inherit their parent's split; repeat leakage checks after normalization.

**Deliverable:** normalized source records, tested mappings, a frozen split manifest, and a leakage/privacy report.

### Stage C — Construct realistic memory-agent invocations

One training example represents one memory-agent invocation, potentially with multiple tool calls:

```text
INPUT
  exact task instructions and tool definitions
  memory actually visible at that point
  permitted source chunk, references, time, and budgets

TARGET
  assistant tool call(s)
  receipts generated by isolated runtime handlers
  final completion / stop
  accepted memory changes and resulting state
```

Store raw evidence separately from model-visible evidence. The teacher must not use future events, hidden tool-output sections, or unavailable prior memory to produce an answer the student could not infer.

Support two explicit modes: correction against a captured/reconstructed historical input, and **fresh causal replay** under a pinned target policy when exact historical calls are unavailable. Never present fresh replay as the original trace.

Build a teacher-clean replay path: accepted teacher state becomes input to subsequent jobs. Recompute identities, support links, budgets, and triggers when memory changes. Preserve the runtime's within-job snapshot behavior instead of inventing a different update schedule.

**Deliverable:** reproducible invocation inputs, provenance, state snapshots, and replay fixtures. Summary-only records remain outside primary source-grounded training.

### Stage D — Produce and validate better labels

Treat existing Qwen/Pi memories as candidate answers, not ground truth.

First, audit the permitted input for important new facts without showing Qwen's answer. Have a stronger teacher independently generate valid actions from that same input. Only then compare candidates and retain, repair, or replace the original output. This allows discovery of facts Qwen omitted instead of merely polishing what it wrote.

A historical Qwen output is a paired candidate only when its input is identical. After repacking or changing prior memory, rerun Qwen on the new input or retain its original answer as diagnostic context only.

Run structural checks, then a blinded semantic judge that examines evidence, omissions, scope, attribution, and unnecessary duplication. Calibrate the judge against human-reviewed good and deliberately corrupted examples. Escalate ambiguous or severe cases; allow only bounded repair attempts.

Generate tool receipts through isolated handlers, never through teacher invention. Store original candidates, changes, decisions, model versions, and costs. Only accepted actions advance a teacher-clean replay; unresolved failures must block or quarantine that dependent trajectory rather than silently advancing it.

**Deliverable:** accepted labels with auditable evidence, plus rejected attempts and a human-review queue. Begin with the specification's proposed 200–300-invocation Observer pilot where the corpus supports it; scale only after reviewing quality and cost.

### Stage E — Release datasets and train candidates

Publish an internal, immutable dataset release containing accepted examples, source/state references, split assignments, policy versions, judgments, and exclusions. Materialize the student's exact chat/tool format from that release.

Before bulk labeling or full training, run a small compatibility test covering tool parsing, multiple calls, receipts, no-op, termination, context limits, and a tiny train/save/reload cycle. Verify actual loss masks: input transcripts and tool receipts are not prediction targets. Do not silently truncate inputs or invent reasoning traces to fit a template.

Run the first Observer LoRA experiment on accepted labels. Keep training settings and artifacts versioned. Use validation data for model and checkpoint selection; keep the final test set untouched by training decisions.

**Deliverable:** reproducible training configuration, export validation, checkpoints/adapters, and training reports. The pipeline must also report incompatibility or insufficient usable data rather than forcing a training run.

### Stage F — Evaluate and decide

Build evaluation before large-scale generation or training so every candidate is measured against the same declared protocol.

| Evaluation | What it establishes |
|---|---|
| **Fixed-input invocations** | Can the model extract the right facts, cite valid evidence, avoid unsupported claims, and complete the tool loop? |
| **Whole-history memory replay** | Does quality survive repeated updates and compactions when the model uses its own accumulated memory? |
| **Downstream reader tests** | Can a fixed assistant recover decisions, constraints, exact details, current state, and unfinished work from the resulting memory? |
| **Operational measurements** | What are latency, memory usage, failure rate, and cost under the intended deployment configuration? |

Compare deployed Qwen, unmodified LFM, fine-tuned LFM, and a stronger teacher reference. A later ablation can compare training on original Qwen labels versus corrected labels.

Judge factual support and important-fact retention separately. Track invented completions, lost user corrections, stale decisions, duplicate growth, and unsafe pruning. Do not let a single average score hide critical failures. Include justified empty outputs and unanswerable downstream questions.

Use the same downstream reader, recent-context suffix, and retrieval policy. Report memory-only results separately from bounded search/recall results. Fixed-transcript replay tests memory behavior, not a fully counterfactual coding session.

Freeze decision criteria before final testing and report uncertainty across independent session families. A candidate may justify replacement through better quality, or comparable quality with a meaningful operational benefit. Lower training loss alone is not sufficient.

**Deliverable:** a reproducible comparison and an explicit adopt / iterate / reject decision, including results for the intended deployment quantization.

### Stage G — Integrate cautiously

Only after evaluation passes, connect the candidate through the normal Hatfield model interface. Start with shadow calls that cannot mutate live memory. Compare their outputs and runtime failures before enabling a limited rollout.

Keep the previous model configuration available for rollback. Production fallback behavior must be explicit and reported; it must not hide candidate failures during offline evaluation.

**Deliverable:** an opt-in model configuration, shadow report, rollout criteria, and rollback instructions.

## 4. Implementation boundaries

Use a small offline pipeline, not a new distributed platform.

**Python orchestration** owns job manifests, provider calls, annotation workflows, dataset releases, tokenization, training, evaluation, and reports.

**A pinned PHP bridge** owns Hatfield-compatible request rendering, tool schemas, isolated action validation, identity handling, and memory projection. Reuse production logic where practical rather than maintaining a subtly different Python implementation.

**Versioned Pi adapters** preserve native session/branch semantics and normalize them into the target contract. Unsupported versions fail visibly.

Keep raw archives, candidate attempts, accepted examples, and model-specific exports separate. Cache model work by exact input, policy, model, and attempt settings. Record retries and spending; enforce configurable limits before bulk calls. Teacher/judge agents receive no shell access or production-write tools. Historical tool calls are evidence, not commands to execute.

The existing specification package supplies proposed schemas, a synthetic example, and structural validation scaffolding. It is a starting contract, not an already implemented extraction/training system.

## 5. Build order and milestones

| Milestone | Build | Required result before proceeding |
|---|---|---|
| **M0: Feasibility** | Inventory sample exports; inspect runtime contracts; test LFM compatibility | Known input formats, limitations, and a viable tool/template/training path. |
| **M1: Data foundation** | Importers, privacy boundary, lineage splits, PHP bridge, invocation construction | A small real corpus can be reconstructed or explicitly fresh-replayed without leakage. |
| **M2: Labeling pilot** | Teacher, validators, calibrated judge, review workflow | Reviewed accepted labels, failure taxonomy, and measured generation cost. |
| **M3: Evaluation baseline** | Fixed-input, chronological replay, and reader-test harnesses | Qwen and unmodified LFM have comparable baseline reports. |
| **M4: Dataset release** | Scale accepted labeling and freeze model-specific exports | Reproducible release with provenance, quality decisions, and exclusions. |
| **M5: Training experiment** | Observer LoRA and held-out comparison | Evidence-based replacement decision. |
| **M6: Expansion** | Shadow deployment; later Reflector/Dropper experiments | No unmeasured broad replacement of the memory pipeline. |

M2 and M3 can develop together. Compatibility, pilot quality, and baseline evaluation are gates before expensive scaling. Numerical targets in the companion specification are provisional and must be calibrated, not treated as measured guarantees.

## 6. First assignment for the implementing agent

Read this plan, then the companion specification and schemas. Inspect the actual Hatfield extension and available session samples, and record the code revisions used. Do not silently replace the specification's pinned behavior with whatever `main` happens to contain.

Implement the **M0/M1 foundation first**: a read-only inventory command, one supported Hatfield importer, one supported Pi importer, and a small set of end-to-end fixtures covering a normal observation, a justified no-op, and a provenance or version failure. Test the target model's tool/template path separately; absent model access should be reported, not simulated as a successful compatibility result.

Return a runnable foundation, fixture tests, an inventory report, explicit unsupported cases, and a task breakdown for labeling and evaluation. Do not start bulk paid labeling, publish private data, launch a full training job, or switch Hatfield's production model as part of this assignment.

**The project's central question is: can a small specialist preserve the memory that matters for future coding work, reliably and efficiently, over an entire session—not merely write plausible-looking observations once?**
