# OM dataset collection

Collect a private source corpus from legacy Pi OM, current Pi OM, and Hatfield sessions. The corpus preserves original histories and memory candidates for later teacher labeling under the Hatfield target contract.

This stage makes no model calls. It does not create accepted training labels, reconstruct exact historical prompts, freeze evaluation splits, or run training.

The [first joint training experiment](hatfield-om-dataset-spec/OM-PLAN-02-judge-guided-training.md) defines the next work: export captured traces, prove student compatibility, review a bounded batch automatically, train one adapter across all three roles, and measure held-out performance. More Jev calibration is not a prerequisite.

## Collect sources

Raw archives and memory candidates may contain credentials and personal information. Keep `data/` private. Git ignores this directory. The collector creates its output directory with mode `0700`, writes data files with mode `0600`, and marks completed archive files read-only. These permissions are not encryption or redaction.

Install the locked dependencies:

```bash
uv sync --locked
```

Inspect the source file counts and sizes without copying or printing session content:

```bash
uv run python main.py inventory \
	--hatfield-om-dir /home/ineersa/projects/sqlite-queue/.hatfield/extensions-data/observational-memory \
	--hatfield-om-dir /home/ineersa/projects/agent-core/.hatfield/extensions-data/observational-memory
```

Collect every Pi session under `~/.pi/agent/sessions` and both selected Hatfield stores:

```bash
uv run python main.py collect \
	--hatfield-om-dir /home/ineersa/projects/sqlite-queue/.hatfield/extensions-data/observational-memory \
	--hatfield-om-dir /home/ineersa/projects/agent-core/.hatfield/extensions-data/observational-memory \
	--output data/collection-2026-10-13-v1
```

Choose a new output directory for each collection. Existing output directories are not overwritten. No live session, database, or configuration is migrated.

Use `--pi-root PATH` to select another Pi session directory. Use `--no-pi` for a Hatfield-only collection. Each Hatfield OM directory supplies its `.sqlite` databases and defaults to the corresponding `.hatfield/sessions` journals. For custom session storage, repeat `--hatfield-session-root PATH` in the same order as the OM directories.

### Resume an interrupted collection

```bash
uv run python main.py collect --resume --output data/collection-2026-10-13-v1
```

Resume uses the stored source inventory, not a new scan. Source-selection flags are ignored. Newly created sessions require a new collection. Completed source captures are reused even if their live files change or disappear. Sources not yet captured must remain accessible. Resume rejects changed collector code or dependencies rather than mixing parser versions.

A completed collection cannot be resumed or extended.

### Reparse a frozen corpus

After a parser fix, create a new derived corpus from the existing archives:

```bash
uv run python main.py normalize data/collection-2026-10-13-v1 \
	--output data/collection-2026-10-13-v2
```

This command verifies and reparses the frozen inputs. It does not rediscover live sessions. The derived corpus preserves upstream source metadata and the original manifest hash. The original release remains unchanged.

## Select the OM dataset

The source archive preserves all collected histories. Use `curate` to create the OM-only dataset view:

```bash
uv run python main.py curate data/collection-2026-10-13-v2 \
	--output data/curated-2026-10-13-v2
uv run python main.py validate data/curated-2026-10-13-v2
```

The view retains only sessions with usable OM records. Observer candidates require valid structure and resolved content references. Reflector and Dropper candidates require references to observations with resolved content evidence. Memory snapshots require matching observation records and resolvable reflection support. Missing journals, invalid ancestry, missing evidence, and ambiguous native identities exclude affected records and their dependent states.

All four relevance levels remain unchanged. There is no filter for low relevance, conversational wording, or session length. Large sessions remain useful sources for reflection and drop behavior.

Deduplication removes only identical native observation records in the same source archive and branch. Repeated text with different evidence or a different branch remains distinct. Exclusions preserve links to retained duplicate records. Pi V3 dependency resolution follows whole-batch validation and first-valid-record-wins before checking content evidence. Legacy identity aliases with conflicting values are not treated as interchangeable.

The view contains `memory-candidates.jsonl`, `sessions.jsonl`, `eligibility.jsonl`, `exclusions.jsonl`, a report, and the curator code used. Its manifest pins the upstream manifest hash. Native archive references resolve in the upstream collection, so retain both directories. Validation verifies the complete upstream partition, original record equality, evidence proofs, and exclusion decisions. The view's `.gitignore` also protects custom output locations.

### Agent definitions and teacher inputs

The corpus preserves native memory records and normalized argument fields. The separate policy export below adds source-pinned agent instructions and worker tools. Complete historical memory-agent requests and receipts remain absent. The proposal's schemas and synthetic tool fixture are not byte-exact production definitions.

Pi-derived training examples use the Hatfield target contract. Historical Pi tool fields remain native provenance rather than being mixed into that contract. Generation uses a Python tool loop and local handlers, not a Hatfield service or PHP bridge. Native Pi definitions remain available for later transfer evaluation; Pi-shaped outputs are not training targets.

Give teacher and student the same permitted source content, visible memory pool, instructions, tools, and limits. Do not add a separate style-cleaning policy. Generate fresh actions under those definitions and obtain receipts from isolated runtime handlers. Exact historical requests can only be reused when they were captured or can be proven reconstructable; an OM ledger entry alone is not that proof.

## Export agent policies

Use a new output directory. The export requires Git, PHP with `nikic/php-parser` available through Composer, and Node with TypeScript and TypeBox installed in the selected Pi runtime repository. It reads committed sources and does not require provider credentials.

```bash
uv run python main.py export-policies data/curated-2026-10-13-v2 \
	--hatfield-repo /home/ineersa/projects/agent-core \
	--pi-repo /home/ineersa/claw/pi-observational-memory \
	--pi-runtime-repo /home/ineersa/claw/pi-mono \
	--output data/policies-2026-10-13
uv run python main.py validate data/policies-2026-10-13
```

`--php-parser-autoload` defaults to the selected Hatfield repository's `vendor/autoload.php`. The Pi revision options accept hexadecimal commit IDs, not branch names. Hatfield uses the corpus's target revision and rejects a target other than the supported pin.

### Policy bundles

| Bundle | Source revision | Agents | Native output contract |
|---|---|---|---|
| `hatfield-target` | `5ea2fd920d04eccee75094afe033302c5596393c` | Observer, Reflector, Dropper | Hatfield worker tools. |
| `pi-current-native` | `58f05fa0a857bc4fcd3dda0b3e845e8bfe069adf` | Observer, Reflector, Dropper | Pi worker tools with native field names. |
| `pi-legacy-v2-reference` | `00d5975ccbb19fcdc2f5e4b04e83688ae439be10` | Observer, Reflector, Pruner | Text completions. No worker tools. |

Each bundle contains `definitions.json` and its committed source files. Definitions include exact instruction strings, tool names, descriptions, parameter schemas, identities, and source ranges. The export archives input builders, runtime expressions, handlers, and supporting implementation code without executing application code. PHP literal extraction uses the parser AST. TypeScript schema extraction evaluates a restricted AST through the installed TypeBox factories. Unknown factory imports and dynamic or shadowed bindings fail extraction. The Pi-AI TypeBox re-export is verified and archived.

Hatfield call options are part of the definitions: Observer permits six tool calls; Reflector and Dropper permit sixteen. Each has a 300-second request limit. Dropper sets thinking to `off`; the other agents inherit the runtime setting.

The manifest pins the corpus manifest hash, source revisions, artifact hashes, and parser versions. The release also preserves the exporter and Python dependency lock. Validation runs without reopening live source repositories. It checks artifact integrity, definition identities, schema structure, source ranges, and archived source expressions. It does not execute handlers or establish semantic label quality.

The historical V2 implementation is a reference, not proven attribution for the enriched `om.observation` records or version-4 snapshots. Their producer remains unresolved. Native definitions from known commits do not prove the exact requests used in historical sessions.

The policy release has `training_ready: false`, `handlers_executed: false`, and zero model calls. It contains source code for dynamic input construction, not reconstructed invocation inputs. User-facing recall and search tools are outside the worker tool set.

## Reconstruct historical Observer ranges

Use a new output directory. This command reads frozen archives through the curated OM-only selection. It does not reopen live sessions or call models.

```bash
uv run python main.py reconstruct-ranges data/curated-2026-10-13-v2 \
	--output data/ranges-2026-10-13
uv run python main.py validate data/ranges-2026-10-13
```

### Range records

`ranges.jsonl` contains only usable ranges and links historical batches to archived journals and retained Observer candidates. Current Pi ranges follow parent ancestry and start after the greatest prior observation coverage boundary. Ranges with conflicting evidence, off-branch endpoints, unknown legacy source selection, or unmatched Hatfield hashes are excluded. `exclusions.jsonl` records their identities and reasons for audit; it is not a replay input. Frozen source archives remain unchanged.

Hatfield ranges come from `om_coverage`, scoped by store and run. The Python renderer tries interval-local blocks, globally paired tool groups, and neighboring block boundaries. It also tries the historical 3.25-character token estimate because the tool-output digest threshold changed to 2.35 without a renderer-version change. A source section is verified only when both `source_digest` and `part_digest` match. Multipart boundaries and unmatched hashes remain flagged, not replaced with guessed inputs.

| Status | Meaning |
|---|---|
| `source_verified` | Hatfield source blocks match both historical hashes. |
| `range_inferred` | Current Pi boundaries follow the known native coverage rule. No request hash is available. |
| `range_only` | Endpoints are recoverable, but source rendering or producer attribution remains uncertain. |
| `unresolved` | The available graph or coverage metadata cannot establish a range. |

The manifest accounts for every upstream Observer candidate across the retained and excluded ranges. Validation checks artifact integrity, upstream manifest hashes, range identities, clean-range selection, counts, and candidate coverage. It does not rerun source reconstruction. Every row keeps `historical_request_identity_established: false`: source hashes do not recover the original visible memory, fallback clock, or complete request.

`pilot-jobs.jsonl` contains three independent first ranges with verified Hatfield source bytes. These jobs start fresh memory rather than claiming to reproduce historical memory. Their fallback time comes from coverage metadata, not a recovered clock. The jobs are development inputs, not approved training examples.

## Run a small Observer pilot

The prototype is one script, `om_dataset/teacher.py`. It reads prepared development jobs and the exported Hatfield Observer definition. It uses an OpenAI-compatible chat-completions endpoint, executes `record_observations` locally, returns progress receipts, and saves the conversation and collected observations. It does not launch Hatfield or execute PHP. Synthetic tool-call and continuation checks passed on both OpenCode Go and direct DeepSeek with low reasoning. These checks do not establish label quality.

Keep jobs and results under private `data/`. Each input JSONL row has `run_id`, `cutoff_seq`, `local_time`, and `sources`. Ordinary sources have an increasing `seq`, `role` (`User`, `Assistant`, or `Tool result`), `timestamp`, and `text`. Tool results may also have `tool_name`. Reconstructed sources set `native_rendered: true` and supply `source_refs`. Their `segments` metadata contains trusted character offsets, speaker, timestamp, and framing from the source builder. Oversized multi-event blocks require these segments; the runner does not infer tool-group boundaries from quoted text. Native emission order can repeat or move backward in anchor sequence, and tool groups include both call and result references. Times use `YYYY-MM-DD HH:MM` in a consistent declared timezone.

Paths and settings are fixed near the top of the script. It reads `data/teacher-pilot/two-session-jobs.jsonl`, loads `data/policies-2026-10-13`, and writes `data/teacher-pilot/two-session-attempts.jsonl`. Each run processes unfinished chunks in the prepared jobs file. Run commands from the repository root.

Prepare one small Pi session and one small Hatfield session from the frozen ranges. The preparer selects two to four consecutive recovered ranges per session, rejects sessions with excluded ranges, and starts each memory stream empty. Pi sources retain visible text and tool calls but omit thinking; Hatfield source sections retain their hash-verified rendering. Uncovered session tails are not invented. This command refuses to overwrite existing jobs:

```bash
uv run python -m om_dataset.pilot
```

Validate inputs without credentials or network calls:

```bash
uv run python -m om_dataset.teacher --dry-run
```

Before a live run, inspect the jobs and authorize their transmission to the provider. The script defaults to OpenCode Go and reads the existing `OPENCODE_API_KEY` environment variable. No model or URL setup is required. Do not put credentials in source files or URLs.

To switch to direct DeepSeek, change `PROVIDER = "opencode-go"` to `PROVIDER = "deepseek"` near the top of `om_dataset/teacher.py`. That route reads `DEEPSEEK_API_KEY`. Both routes use DeepSeek V4.1 Flash with thinking enabled and `reasoning_effort: "low"`. OpenCode Go uses model ID `deepseek-v4.1-flash`; direct DeepSeek uses its `deepseek-flash` alias. [DeepSeek's thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode) maps low to low, not high.

The OpenCode Go transport follows Hatfield's `OpenCodeSessionHttpClient`: it sends `x-opencode-session` using the job's `run_id`, plus `x-opencode-client` and `User-Agent` identifying this script as `om-project`. The session ID stays unchanged across tool calls and continuation requests. Direct DeepSeek does not receive OpenCode-specific headers.

Run the pilot:

```bash
uv run python -m om_dataset.teacher
```

Requests use a fixed `max_tokens: 65536` budget for reasoning and response tokens. There are no provider, model, path, or request-option CLI flags. Productive runs have no tool-call-count cutoff. `RUN_LIMITS` bounds each attempt to 900 seconds and 131,072 cumulative completion tokens. Three consecutive tool-response turns with no newly accepted observation end the attempt as `stalled`. Elapsed time, completion-token exhaustion, and context exhaustion produce `budget_exhausted`, not successful no-ops. HTTP timeouts respect the remaining attempt deadline.

Source slicing uses LFM's 131,072-token student window, not the DeepSeek teacher's 1M capacity. After reserving 65,536 output tokens, initial requests use 65% of the remaining input allowance, approximately 42,598 estimated tokens. The remaining input space allows tool-loop growth. The estimator counts serialized messages and tool definitions at 2.35 characters per token, plus framing overhead. The runner checks context plus output allowance before every provider call, including transient reasoning content. It also checks the final saved sequence without reasoning. Results report estimates and `exact_tokenizer_checked: false`; actual student chat-template and tokenizer validation remains required at SFT export.

Packing groups complete source blocks when they fit. Oversized blocks split at Unicode character boundaries within trusted segments. Every native fragment repeats its speaker, original timestamp, tool framing where applicable, and an explicit continuation marker. Original text and offsets remain available. No source text is truncated. Visible prior memory receives up to 55% of the initial envelope after fixed overhead. The prototype trims observation lines from the front, then reflection lines from the end; it does not reproduce Hatfield's timestamp/relevance sorting. The original jobs file remains unchanged.

### Credential redaction

The runner replaces known credentials with `[REDACTED]` and padding before slicing or sending source text. It also cleans prior memory, model responses, tool arguments, and reasoning before saving checkpoints or accepting observations. Saved traces contain the redacted content. Masks preserve character counts and line breaks, so trusted source segments and continuation offsets remain valid. Provenance identities and archive references must not change; a credential in those fields stops the run.

Live labeling requires the private catalog at `.hatfield/security/redaction-values.json`. It contains detected credential values, has mode `0600`, and is Git-ignored. Never include it in a training export. The runner also checks current environment credentials, common token formats, private keys, authentication headers, URLs with passwords, and password assignments. To create the catalog on a new checkout with a local GitGuardian report:

```bash
uv run python -c 'from om_dataset.redaction import build_catalog; print(build_catalog())'
```

Catalog creation refuses to overwrite an existing catalog. Each job records a fingerprint of the redaction rules and catalog values. Environment credentials are also masked, but rotating them does not change that fingerprint or invalidate completed work. Resume refuses existing traces or checkpoints that still contain detected credentials. Frozen archives and historical candidate records remain unchanged for provenance. They are private source material, not sanitized training exports. Use the runner's redacted messages and labels for training.

### Causal state and resume

Jobs advance chronologically within each `(run_id, state_scope)` stream. Historical jobs require an explicit `state_scope` that separates archives and branches. The pilot uses store identity plus run ID. Reversed or repeated cutoffs fail visibly; branch inheritance and chronological job scheduling are not inferred automatically. Optional `state_parent_id` asserts the expected prior state. Initial `prior_observations` and `prior_reflections` seed a stream; subsequent jobs receive its teacher-generated state instead of loading historical labels.

All parts of one job share the same pre-job memory snapshot. Only after every part completes does the runner merge their accepted observations into the next job's memory. Saved jobs include `state_before_id`; the final successful part records `state_after`. A failed, stalled, or exhausted part prevents state commitment and blocks later consumers in that stream. Other streams remain independent. This is Observer-only state advancement, without reflection or dropping.

No automatic network retries occur. Rerun the command to continue unfinished work; completed chunks with identical input, policy, configuration, and runner code are skipped. Atomic private checkpoints in `attempts.jsonl.checkpoints/` save accepted observations, messages, usage, and progress counters after every completed tool round. Resume continues from that round rather than restarting the chunk. A final response is checkpointed before appending its result, so an interrupted append does not require regenerating it. Checkpoints may retain provider-required reasoning solely for continuation; they are not training records and are deleted after the result is durable. A response interrupted before it is received and saved must still be requested again. Resume reconstructs memory from successful chunk results before building later inputs.

Concurrent writers are rejected. An incomplete final JSONL record without a newline is quarantined in a private `.torn-<hash>` file, then the valid prefix is resumed. Corrupt complete lines or corruption before the tail fail visibly.

Outputs have `quality_status: unreviewed` and `training_ready: false`. Invalid references and provider failures do not become successful no-ops. An empty Reflector or Dropper batch followed by a final no-action response can finish without changing memory; its rejected call remains in the trace with `protocol_recovery` and required recovery masking. It is not a clean training target. Provider-required `reasoning_content` is passed during continuation but excluded from saved training messages. Results include token usage when the provider supplies it; dollar costs are not calculated.

This is a fresh-labeling prototype with recovered source sections, not exact historical requests or a complete Hatfield runtime port. Student-sized slicing can subdivide a recovered range; it does not merge separate historical invocations. The Observer checks schema and calendar validity more strictly than the native handler, preserves partial acceptance and duplicate receipts, and uses the pinned observation identity formula. Its packing is not byte-identical to Hatfield's. The command above runs Observer only. It does not judge semantic quality, freeze splits, or export SFT token masks. Supplied initial memory requires separate causal review. The small pilot outputs remain unreviewed, not approved training examples.

## Run the larger memory pilot

The larger replay uses the same tool loop for Observer, Reflector, and Dropper. It writes separate files under `data/teacher-long-pilot/`; it does not replace the small pilot, review cases, identity mapping, or votes.

Prepare and validate the two histories without provider calls:

```bash
uv run python -m om_dataset.long_pilot --dry-run
```

Run the replay, or rerun the same command to resume:

```bash
uv run python -m om_dataset.long_pilot
```

`jobs.manifest.json` records the selected histories, input hash, source sizes, and thresholds. Selection prefers substantial clean prefixes from session start. Pi markers must follow both coverage links and parent ancestry. Hatfield sections must retain both historical hash proofs and contiguous coverage. The preparer never skips a broken boundary or invents an uncovered tail. Historical observation sizes select the cohort; historical memories never enter teacher inputs. Reported initial chunk counts assume empty prior memory; actual slicing can change as teacher memory grows.

### Maintenance and recovery

After an Observer range commits, reflection runs when the active observation content exceeds 40,000 estimated tokens. Dropper runs only when reflection adds at least one new record and the pool exceeds 30,000 tokens. These are the frozen Hatfield defaults, not lowered test thresholds. The server caps and ranks proposed drops by coverage, relevance, timestamp, and proposal order. The cap is not a requirement to drop that many records or a guarantee of reaching the target.

Reflector and Dropper see request-local decimal observation IDs. Accepted records use canonical IDs. Existing reflections remain active; dropping an observation does not remove its reflection support link. Oversized maintenance pools are partitioned without discarding observations. Traces identify each visible partition and any reflections omitted from that request's context.

Reflection and drop tool traces are saved as pending generation work. A durable `generation_commit` row applies new reflections and selected drops together. Its state hash becomes the next Observer range's parent. A failed or interrupted maintenance phase cannot publish a partial generation. Completed phases and tool rounds are reused on resume; dependent ranges remain blocked until the generation completes.

`attempts.jsonl` contains agent traces and generation commits. Progress logs print counts and boundaries, not conversation text. `run.json` pins input, policy, provider settings, redaction policy, and runner code before the first live call. Changed code or settings fail visibly instead of silently restarting paid work in the same campaign. Transport credential rotation remains compatible with resume. Outputs still require human review before training.

### Resume the existing campaign after the maintenance fix

The original campaign is pinned to the earlier runner. Do not edit its `run.json` or restart it with changed code. Use the recovery entry point:

```bash
uv run python -m om_dataset.recover
```

Recovery writes to `data/teacher-long-pilot/recovered/state/`. It verifies that the runner changed only at the empty-maintenance final-response guard, reproduces every imported input and tool schema, and retains the original log and pin. Compatible successful traces receive explicit cache aliases. Saved failed replies are revalidated offline through the fixed loop; their visible traces must remain identical, their deltas must stay empty, and their rejections remain flagged. This import makes no provider calls. The replay reconstructs generation commits and requests only unfinished work. The recovery output has its own code and input pin.

## Run controlled memory comparisons

Prepare six fixed-input cases from Hatfield histories only: two Observer, two Reflector, and two Dropper invocations. Pi-derived cases are excluded from judge labeling.

```bash
uv run python -m om_dataset.comparison --dry-run
uv run python -m om_dataset.comparison
```

The comparison reuses captured DeepSeek replies and runs the deployed local Qwen endpoint on exactly the same initial messages, visible memory, original instructions, and tool schemas. No brevity instruction is added. Both runs use the 131,072-token student window and 65,536-token output allowance. The local server's model identity and default generation settings are recorded privately. Reasoning controls differ by provider, so this is not an equal-compute benchmark.

Selection uses manageable nonempty invocations from committed maintenance generations and excludes source tasks with failed attempts. The six cases form a calibration sample, not an unbiased win-rate estimate or two independent end-to-end replays. Reflector evidence is its visible memory pool. Dropper candidates are proposals for the visible partition, not the entire generation's committed removals.

Private snapshots, checkpoints, results, and selection exclusions live under `data/teacher-comparison/`. Preflight records remain separately under `data/teacher-comparison-preflight/`. The exporter stages fixed cases at `data/review-next/cases.jsonl` and the producer mapping at `data/review-next-private/identity-map.json`; it refuses to overwrite or reshuffle either. Candidate text displays accepted actions once, with supporting IDs and any rejected attempts. Raw traces remain in the private results.

For the existing mixed-source pack, replace its three Pi cases without changing the three Hatfield cases:

```bash
uv run python -m om_dataset.hatfield_review
```

Replacement inputs, results, and checkpoints live under `data/teacher-hatfield-comparison/`. The command excludes previously selected source jobs, reuses captured teacher replies, and generates three new local-model replies. It publishes only after all three invocations complete. Existing Hatfield wording, A/B assignments, and votes remain unchanged. Pi votes and the old mixed-source packs move to `data/review-history/`; they do not count toward the active judge-labeling pack. Rerunning resumes generation or an interrupted publication without reshuffling candidates or overwriting later votes. The ordinary comparison command refuses a legacy mixed-source manifest rather than silently reusing its Pi cases.

## Review candidates in Hatfield

The project skill `.hatfield/skills/om-review/SKILL.md` runs in the main session. It presents blinded cases, asks for a human preference through `ask_human`, saves each decision atomically, and resumes from the next unanswered case. It does not delegate review to a subagent, run a judge, or supply votes itself. The skill file is trackable; review data, identity mappings, and runtime files remain private.

The active controlled pack contains six Hatfield-only cases covering all three roles. Historical-content comparisons and retired mixed-source packs remain under `data/review-history/` for provenance, not active judge labeling. Replacing Pi cases preserves votes on unchanged Hatfield cases and does not prefill votes for replacements. The facilitator's workflow is unchanged.

The controlled pack uses matched captured inputs. Extra observations alone do not establish an improvement.

Start a fresh review session from the repository root:

```bash
hatfield --skills om-review
```

Ask to start or resume the review. Within an existing session, `/skill:om-review` loads the same workflow in the main session, but a fresh session avoids contamination from earlier generation discussions.

Cases live in `data/review/cases.jsonl`. Votes live in `data/review/votes/<case_id>.json`, with a case hash, human choice, optional explanation, timestamp, and revision history. Choices are A, B, both acceptable, neither acceptable, or skip. Discussion and cancellation do not become votes. The facilitator must not read `data/review-private/identity-map.json` or infer producer identities. This is procedural blinding, not a filesystem sandbox. No human review decisions are prefilled.

## Try the Jev judge

The Jev adapter uses TypeSafe's native `Choice` questions, not chat completion or prompted JSON. It pins `jev-1.13.0` and applies [the memory quality rubric](hatfield-om-dataset-spec/om-jev-quality-rubric.md) to six blinded review cases and twelve synthetic controls. Each pair runs in both A/B orders, for at most 36 requests. `--tasks reflector dropper` selects four review cases and seven controls, for 22 requests.

Set `JEV_API_KEY` or `TYPESAFE_API_KEY` in the shell environment. If the key is in your Bash configuration, load it with `source ~/.bashrc`. Do not pass credentials as command arguments. The existing private redaction catalog is also required.

Inspect the inputs and approximate context sizes without API calls:

```bash
uv run python -m om_dataset.jev_judge --dry-run
uv run python -m om_dataset.jev_judge --tasks reflector dropper --min-confidence 0.85 --dry-run
```

Before a live run, authorize sending redacted evidence to TypeSafe. Test the first synthetic pair. Use a fresh output directory when code or inputs have changed. This focused batch includes a case that requires long context:

```bash
uv run python -m om_dataset.jev_judge --controls-only --max-new-calls 2
uv run python -m om_dataset.jev_judge --tasks reflector dropper --min-confidence 0.85 --long-context --output data/jev-judge/v5
```

### Quality checks

The v4 adapter keeps the five quality checks introduced in v3:

| Worker | Checks |
| --- | --- |
| Observer | Useful coverage, context recoverability, importance assignment, compression |
| Reflector | Useful coverage, context recoverability, compression |
| Dropper | Safe removal |

Each check returns `pass`, `needs_work`, or `uncertain`. Importance assignment evaluates the observation's existing relevance tag, not the severity of a judge-detected defect. A speculative location question selects an affected candidate record or missing source fact. Code ignores that selection unless the check needs work. A failure without a location becomes uncertain rather than a confirmed defect.

The full original role instructions and actual model-facing tool schemas remain in every request. Internal PHP handler source and runtime bindings are excluded because the worker did not receive them. Source evidence and existing memory are not truncated. The original [six controls](hatfield-om-dataset-spec/fixtures/judge-controls.json) remain unchanged; [two additional controls](hatfield-om-dataset-spec/fixtures/jev-quality-controls.json) test identifiers and assigned importance. [Four Reflector and Dropper controls](hatfield-om-dataset-spec/fixtures/jev-reflector-dropper-controls.json) test preserved critical facts, deletion of the last copy, current-state revisions, and valid selective no-ops.

The v4 prompt changes apply only to Jev's judging criteria. Reflector checks distinguish unreflected observations from lost information and stale claims from missing identifiers. Dropper checks require comparing a proposed deletion with every existing reflection and kept observation, after all proposed deletions together. Worker prompts, candidate memories, and Observer judging requests remain unchanged.

The pair gets an independent preference question. Compression cannot override failed coverage, recoverability, importance, or safe removal. Both candidates can preserve the facts while the more compact candidate wins. Raw selections, probabilities, confidence, and preference adjustments remain recorded. Jev does not generate explanations, corrected memories, or quotations. Human votes and producer mappings never enter requests.

### Results and resuming

Results remain private under `data/jev-judge/`. The focused v4 trial is under `data/jev-judge/v4-focused/`; `data/jev-judge/v4/` preserves the initial partial run. The focused runner revalidates six identical recorded replies and makes sixteen new calls. It uses normal context except for `memory-006`, whose stricter request exceeded the normal-context limit. Its `run.py`, `audit.py`, and `notes.md` record the procedure and limits. Earlier results are not overwritten. The standard CLI does not reuse replies across directories.

Records retain each check's status and location, redacted request snapshots, and typed responses. The adapter validates answer IDs, option sets, probability ranges, and the pinned model version. It allows bounded sum error from two-decimal probability rounding without changing returned probabilities. Non-maximal choices in unused speculative location answers do not invalidate the primary decisions. Active concrete locations and primary answers still require maximum-probability choices.

`--min-confidence` defaults to `0.85`. Primary checks below the cutoff become `uncertain`, not failures or passes. Pairwise preferences below the cutoff abstain too. An identical-candidate override can establish a tie without the pairwise answer. Speculative locations retain their separate confidence; the primary cutoff does not certify a record pointer. Records save both ungated `assessment` and `confidence_assessment`, and reports compare their counts. The cutoff is pinned with the run settings.

`review_status` summarizes all applicable checks, including compression. The legacy `verdict` field summarizes only non-compression checks so earlier factual-adequacy controls remain comparable. A verbose candidate can have `verdict: acceptable` but `review_status: needs_work`. Neither field approves a training example. Reports retain raw and code-composed preferences, check-status and location stability, legacy control results, and separate targeted quality-check results. Status counts include both candidates in every call and both A/B orders. `calibrated` and `training_ready` remain false. Stronger-judge checks and an independent audit are still needed before automated training-data selection.

Rerun with the same options to resume missing requests without repeating completed or failed calls. Service errors stop the batch. Interrupted calls remain failures rather than triggering an automatic retry. Changed code, criteria, inputs, or cutoff require a fresh directory.

Rebuild a complete run's report without API calls:

```bash
uv run python -m om_dataset.jev_judge --tasks reflector dropper --min-confidence 0.85 --long-context --output data/jev-judge/v5 --report-only
```

Context estimates use the project's character heuristic, not Jev's tokenizer. Estimates above the default limits produce warnings; TypeSafe validates actual token counts. The adapter never truncates evidence or automatically retries with larger context. Use `--long-context` only when needed to opt into TypeSafe's slower, less accurate 262k mode. Its default limits are 32k per question and 64k per request. See [TypeSafe's model limits](https://docs.typesafe.ai/models).

## Try the prompted judge

This earlier DeepSeek pilot remains available for comparison. It does not use schema-enforced output; use the Jev adapter above for native typed judgments. The shared prompt is now version 5, so new DeepSeek runs need a fresh directory, such as `--output data/judge-pilot/v6`; earlier results cannot be resumed with the changed prompt.

The judge pilot uses DeepSeek V4.1 Flash with high reasoning through the provider selected in `om_dataset.teacher`. It defaults to OpenCode Go. It uses the six active review cases and six synthetic controls, with each pair evaluated in both candidate orders. This is a development trial, not a calibrated acceptance gate or a training-data exporter.

The prompt treats substantially less memory text and fewer records as primary preferences when the core information survives. Some loss of secondary detail is acceptable. Both candidates can remain acceptable while the more compact candidate wins. Three controls check this trade-off: three observations versus ten, one sentence versus three, and a shorter candidate that must lose because it omits an active approval constraint. These controls supplement the original false-completion, invented-reflection, and unsafe-removal cases.

Inspect [the judge prompt](hatfield-om-dataset-spec/om-judge-prompt.md), then validate inputs without network calls:

```bash
uv run python -m om_dataset.judge --dry-run
```

Before a live run, authorize transmission of the redacted case evidence to the configured provider. The private credential catalog is required. Predictions receive task policies, evidence, and anonymous candidate text. They do not receive human votes, expected control answers, or producer mappings.

```bash
uv run python -m om_dataset.judge
```

The pilot makes at most 24 calls, with 32,768 completion tokens and a 900-second timeout per call. Use `--max-new-calls 2` to run a small part and resume later. It validates response structure and exact evidence quotations. Two logged formatting repairs are permitted: remove unexpected candidate-level fields containing only null or an empty string, and resolve a nested evidence label when its exact quotation is verified within that labeled segment of one outer block. Original reply content is retained; repairs cannot change judgments or quotations. Other invalid assessments, provider failures, and truncated replies remain failures, not successful no-ops or approvals. Provider-required reasoning is not saved. Dollar costs are not calculated.

Results and the comparison report remain private under `data/judge-pilot/v5/`. Earlier trials remain in separate directories. Their predictions cannot be reused with this revised prompt. Each prediction is saved separately, so rerunning reuses saved records. Failed records remain failed; there are no automatic retries. Changed prompts, inputs, configuration, or runner code require a new directory, such as `--output data/judge-pilot/v6`. After a parser-only change, `--reuse-predictions-from data/judge-pilot/v5` revalidates recorded replies in a new directory offline where requests match exactly. Reuse rejects changed prompts, cases, policies, or provider settings and records the original file hash; it does not overwrite the earlier run.

Human votes are loaded after the inference phase and never enter prediction requests. The report compares preferences with those votes, checks order stability, and scores the synthetic controls separately. It reports missing and failed calls separately. Structured swapped verdicts are mapped back to the original labels; free-text explanations retain the as-shown labels recorded in `explanation_label_order`. A preferred candidate is not automatically an acceptable training target. The report always leaves `calibrated` and `training_ready` false.

Rebuild a report from complete saved prediction files without model calls:

```bash
uv run python -m om_dataset.judge --report-only
```

Inspect disagreements and invalid replies before changing the prompt. Keep this six-case pack as development material. Check the revised prompt on a separate human-reviewed batch before trusting automatic acceptance. The existing votes have no per-case written reasons. The reviewer has since clarified that substantially fewer records and less text can outweigh a small loss of secondary detail; preference agreement alone still cannot establish that the judge applies this trade-off correctly.

## Verify a collection

```bash
uv run python main.py validate data/collection-2026-10-13-v2
uv run pytest -q
uv run python hatfield-om-dataset-spec/validate_fixture.py
```

Collection validation checks the manifest schema, artifact hashes and sizes, candidate identities, and candidate counts. The proposal fixture validator checks its synthetic example separately. Neither command establishes semantic label quality or production handler compatibility.

## Corpus layout

| Artifact | Contents |
|---|---|
| `manifest.json` | Format version, pinned Hatfield target revision, artifact hashes, and completion marker. |
| `collection-config.json` | Frozen source inventory, collector fingerprint, and Python version. |
| `archives/` | Content-addressed original JSONL byte prefixes and consistent SQLite backups. |
| `parsed/` | Native headers, source indexes, ancestry, memory candidates, and parser issues per source. |
| `jobs/` | Captured source boundaries and completed parse references for resumability. |
| `collector/` | Collector code, schema, dependency declaration, and lockfile used for this release. |
| `sessions.jsonl` | Session archive references, including explicit database-only runs. |
| `memory-candidates.jsonl` | Unreviewed Observer, Reflector, Dropper, and memory-snapshot candidates. |
| `issues.jsonl` | Malformed records, unsupported formats, missing evidence, and ancestry problems. |
| `report.json` | Structural counts and unresolved limitations. No conversation text. |

The collection uses `om-corpus/1`, not the proposal's training-invocation schema. It sets `training_ready: false`, `redacted: false`, and `split_status: unassigned`. Candidate validity describes structural checks, not factual correctness, exact historical input identity, or acceptance for training.

### Source mapping

- Pi keeps native session IDs, entry IDs, parent links, memory IDs, and evidence IDs. Its normalized source reference is `run_id = "pi:" + session_id`, with `seq` equal to the 1-based physical JSONL line number. Headers and malformed lines count toward that number. This is an addressing convention, not a chronological replay policy.
- Pi source indexes retain all branches. The default ancestry chain does not apply compaction, reconstruct historical worker inputs, or prove call-time availability. Replays must use the native branch and compaction semantics.
- The legacy Pi parser supports structured `om.observation` records, older prose summaries, and memory snapshots with `observational-memory` versions 2 and 4. Missing per-record fields remain missing. The current parser supports the three ledger event types and `om.folded` version 1 snapshots.
- Hatfield keeps its native `run_id` and `seq`. SQLite backups include committed WAL transactions. Backup and journal capture are not one atomic snapshot; unresolved references are reported.
- Observation arguments use Hatfield field names: `timestamp`, `content`, `relevance`, and `source_refs`. Native memory IDs remain provenance. They are not substituted for production-generated canonical IDs. Reflection support IDs and drop IDs remain native until a later target-state mapping step.
- Compaction snapshots remain snapshots. Their observations are not counted as additional Observer invocations. A missing Pi ledger entry is not labeled as a successful no-op.

All histories remain in the archive, including sessions without OM outputs and records with unresolved evidence. Unknown or malformed records appear in the issue report instead of silently becoming valid examples.

## Next stages

1. Establish lineage groups and invocation-level duplicate checks on the OM-only view, then freeze train, validation, and test assignments before training. Grouping prevents shared histories from entering both training and evaluation; it does not exclude long sessions or block label generation.
2. Construct causal Hatfield-contract invocations using the policy bundles, preserving raw evidence separately from model-visible evidence and classifying reconstruction fidelity. Map native memory identities into the target state and its request-local IDs.
3. Inspect the Observer and full memory pilots before scaling label generation. Calibrate evaluation against human reviews. Keep Hatfield-only training and report native Pi transfer evaluation separately on held-out families.
4. Before remote labeling, check provider permissions and the data that may leave the machine. Run the selected teacher with the same input and contract as the student. Validate actions through isolated local handlers and review labels before exporting training examples. Local unredacted collection does not authorize sending secrets to a provider.

The design remains in [`hatfield-om-dataset-spec/OM-DATA-01.md`](hatfield-om-dataset-spec/OM-DATA-01.md). Model compatibility, token counts, family counts, and training readiness are not measured by this collector.
