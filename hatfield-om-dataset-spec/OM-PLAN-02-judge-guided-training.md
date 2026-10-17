# OM-PLAN-02: First joint memory-writer training experiment

**Status:** proposed execution plan, updated 2026-10-16. No training or new paid labeling is authorized by this document.
**Deliverable:** one shared LoRA checkpoint for Observer, Reflector, and Dropper, plus a held-out comparison against its untrained base. A negative result still completes the experiment.

## Fixed decisions

- Keep existing candidates. Correct only demonstrated defects; do not regenerate the corpus.
- Train all three roles together. Do not substitute Observer-only training.
- Use `LiquidAI/LFM2.5-2.6B` for the first compatibility check and training attempt. Stop on a concrete incompatibility rather than starting another model comparison.
- Use the existing DeepSeek high-reasoning judge configuration for a bounded automated review. Use the existing local Qwen configuration for conditional corrections. Pin exact settings before calls. Neither model is assumed correct because of its identity.
- Freeze Jev's current prompts and 85% tagging policy. Jev is optional diagnostic metadata, not an acceptance gate or training reward.
- Do not require manual dataset curation. Automatically skip unresolved cases.
- Do not implement RL, on-policy token distillation, another judge taxonomy, or another calibration campaign before the first result.

This plan replaces the older Observer-only and manual-curation prerequisites for this bounded feasibility experiment. Preserve the privacy, source visibility, tool-contract, and split requirements in [OM-DATA-01](OM-DATA-01.md). Machine-reviewed experimental targets are not a certified training release or gold labels. Keep them separate from approved exports. Paid calls, training compute, and deployment still require their own authorization.

## What exists now

The read-only inventory of the two existing teacher replay files finds:

| Input-verified Hatfield candidates | Count |
| --- | ---: |
| Observer | 135 |
| Reflector | 25 |
| Dropper | 25 |
| Total | 185 |

These are candidates, not approved training examples. Empty outputs still need no-op and rejected-call checks. Final trace tokenization, loss masks, and semantic acceptance are not established.

The candidates occupy two captured streams. Only one stream contains all three roles. Streams are not proved independent history families. Six comparison alternatives do not create six independent histories. A valid unseen-history evaluation must not be manufactured by splitting neighboring invocations from the same family.

Private evidence: `data/training-feasibility/input-inventory.json`. Reproduce the inventory without model calls:

```bash
uv run python - < data/training-feasibility/input_inventory.py
```

The repository has collection, captured replay traces, local handlers, comparisons, and judge pilots. It does not yet have the SFT exporter, tokenizer/mask validation, training runner, or a trained checkpoint. Those are the implementation gaps. Judge calibration is no longer the next milestone.

## 1. Export and prove compatibility before paid labeling

1. Reuse captured inputs and existing policy/handler facilities. Export original instructions, tools, visible memory, source, assistant actions, receipts, and stopping behavior. Do not train on detached memory prose.
2. Reject invalid or unresolved traces. Exclude rejected-action recovery traces from this first experiment rather than building a recovery-mask subsystem.
3. Render the student's real chat template. Check complete-trace token counts, tool-call encoding, and assistant-only loss masks. Do not silently truncate sources or memory. Report usable and excluded counts per role.
4. Run a disposable synthetic train/save/reload microtest. Verify actual parameter updates and tool parsing. This is a plumbing test, not a candidate model or evidence of memory quality.
5. Establish history-family groups. Reserve an independent source history for evaluation before selecting training inputs. Held-out source history does not need existing teacher targets. Keep judge-development cases out of any final-test claim.

**Output:** an export manifest, per-role eligibility counts, decoded target/mask samples, and the compatibility result.

**Stop:** report the exact blocker if the student cannot represent the contract, no usable trace exists for a role, or independent evaluation material cannot be established. Do not respond by tuning Jev or silently changing the task scope. Compatibility work may finish without making a generalization claim.

## 2. Review one bounded batch and train once

1. Select at most ten eligible inputs per role, at most thirty total. Freeze the selection and split before calls. This is a feasibility batch, not a claim that thirty examples establish model quality.
2. Ask the stronger reviewer to judge the existing output against its source and worker contract. Use three outcomes: keep unchanged, correct a concrete defect, or skip because the evidence is unresolved. Check behavior on the existing seeded defects without starting another prompt-tuning campaign.
3. For a concrete defect only, request one correction from local Qwen using the same permitted input. Re-execute its proposed actions in isolated handlers and obtain real receipts. Do not restore facts from hidden history.
4. Review the corrected output in a separate pass. Accept it for this experimental batch or skip it. Do not retry indefinitely or create a manual review queue.
5. Train one shared LoRA on the resulting mixed-role targets. Sample roles deliberately and include valid no-actions where available. Supervise worker actions and stopping, not judge decisions, reasoning, or tool receipts.
6. Save the dataset hash, exact base checkpoint, adapter, template, masks, settings, and training log. Keep original candidates immutable. Corrections to individual targets do not rewrite the captured inputs of later invocations.

**Bound:** at most thirty initial reviews, thirty conditional correction calls, and thirty rechecks. Authorize the provider and compute budget before starting. No automatic expansion to all 185 candidates.

**Output:** counts of unchanged accepted, corrected accepted, skipped, and excluded examples per role; one reloadable adapter. A reviewer acceptance is provisional machine labeling, not proof of correctness.

**Stop:** if a role has no accepted targets, report that gap instead of training only Observer. If the review fails the known critical-defect checks, report the reviewer failure rather than launching another calibration loop.

## 3. Measure the result and decide whether to scale

1. Compare the untrained base and adapter on fixed-input tests for each role. Use deployed Qwen as a reference, not an assumed upper bound.
2. Replay the independent held-out history. Let each model perform all three roles on its own accumulating memory, with fixed policies, budgets, and maintenance rules.
3. Test a fixed reader using source-grounded questions about decisions, current state, constraints, identifiers, and unfinished work. Freeze questions and reference evidence before model comparison. Measure retention before and after reflection and committed drops.
4. Report reader correctness, observed critical failures, tool failures, memory size, and runtime. Keep automated scoring limitations explicit. Lower training loss or agreement with Jev is not the success criterion.

**Output:** one comparison report, including regressions and inconclusive results. This completes the first experiment; it does not authorize deployment.

If the result warrants expansion, review the remaining eligible candidates with the same frozen process, reusing completed work. If student-generated states expose recurring mistakes, collect corrections from training-history replays and run another SFT round. Do not change the first held-out result into training data while continuing to call it held out.

## Hy-Memory connection

Borrow two ideas: evaluate downstream memory usefulness and later train on states the student actually encounters. The first run is ordinary joint SFT, not Hy-Memory's on-policy distillation or RL. Add neither until an exported dataset, trained checkpoint, and held-out result exist.

## Next implementation task

Build the complete-trace SFT exporter and compatibility checks in step 1 using the existing captured inputs. Do not produce another judge-pilot report as a substitute. Report an artifact or a specific failed check, not another open-ended plan.
