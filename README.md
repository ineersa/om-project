# OM dataset collection

Collect a private source corpus from legacy Pi OM, current Pi OM, and Hatfield sessions. The corpus preserves original histories and memory candidates for later teacher labeling under the Hatfield target contract.

This stage makes no model calls. It does not create accepted training labels, reconstruct exact historical prompts, freeze evaluation splits, or run training.

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

Keep jobs and results under private `data/`. Each input JSONL row has `run_id`, `cutoff_seq`, `local_time`, and `sources`. Ordinary sources have an increasing `seq`, `role` (`User`, `Assistant`, or `Tool result`), `timestamp`, and `text`. Tool results may also have `tool_name`. Reconstructed sources set `native_rendered: true` and supply `source_refs`; their text already contains native source labels. Native emission order can repeat or move backward in anchor sequence, and tool groups include both call and result references. Times use `YYYY-MM-DD HH:MM` in a consistent declared timezone. Optional `prior_observations` and `prior_reflections` contain rendered memory lines. Only explicitly supplied memory is visible; historical labels are not loaded automatically.

Paths and settings are fixed near the top of the script. It reads `data/ranges-2026-10-13/pilot-jobs.jsonl`, loads `data/policies-2026-10-13`, and writes `data/teacher-pilot/attempts.jsonl`. The old arbitrary-prefix jobs are no longer the default. Each run processes all unfinished chunks in the prepared jobs file. Run commands from the repository root.

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

Requests use a fixed `max_tokens: 65536` budget for reasoning and response tokens. There are no provider, model, path, or request-option CLI flags. The teacher has no tool-call-count cutoff: it continues until the model finishes or a request fails. This deliberately differs from Hatfield's native six-call setting.

Source slicing uses LFM's 131,072-token student window, not the DeepSeek teacher's 1M capacity. Hatfield's 65% ratio gives an 85,196-token input envelope. Reserving the requested 65,536 output tokens reduces that envelope to 65,536 estimated input tokens, including overhead. The estimate remains Hatfield's `ceil(characters / 2.35)`. Packing accounts for instructions, tools, supplied memory, source headers, and framing. It groups complete source blocks when they fit and splits oversized text at Unicode character boundaries, preserving source IDs and recording text offsets. No source text is truncated or rejected for crossing an arbitrary character count. Each chunk is saved and resumed separately. This sizes initial requests; fitting complete multi-call training traces still requires the student's tokenizer at SFT export.

Visible prior memory receives up to 55% of the envelope after fixed overhead. The prototype trims supplied observation lines from the front, then reflection lines from the end; it does not reproduce Hatfield's timestamp/relevance sorting because the job format contains rendered lines, not native memory objects. The original jobs file remains unchanged. Parts of one job share its supplied memory snapshot; generated observations do not automatically become memory for later jobs.

No automatic network retries occur. Rerun the command to retry failed chunks; completed chunks with identical inputs, policy, configuration, and runner code are skipped. Interrupted chunks may require another provider call. Concurrent writers are rejected, and malformed output logs fail rather than being silently repaired.

Outputs have `quality_status: unreviewed` and `training_ready: false`. Failed jobs never become successful no-ops. Rejected calls remain in the trace and are marked as requiring recovery masking. Provider-required `reasoning_content` is passed during continuation but excluded from saved training messages. Results include token usage when the provider supplies it; dollar costs are not calculated.

This is a fresh-labeling prototype with recovered source sections, not exact historical requests or a complete Hatfield runtime port. Student-sized slicing can subdivide a recovered range; it does not merge separate historical invocations. The runner checks schema and calendar validity more strictly than the native handler, preserves partial acceptance and duplicate receipts, and uses the pinned observation identity formula. Its packing is not byte-identical to Hatfield's: split parts retain source headers, and memory selection uses supplied line order. It does not automatically advance memory between jobs, implement Reflector/Dropper, judge semantic quality, freeze splits, or export SFT token masks. Supplied prior memory requires separate causal review. Pilot jobs with unassigned splits are development material, not training or final evaluation examples.

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

1. Establish lineage groups and invocation-level duplicate checks on the OM-only view, then freeze train, validation, and test assignments before labeling. Grouping prevents shared histories from entering both training and evaluation; it does not exclude long sessions.
2. Construct causal Hatfield-contract invocations using the policy bundles, preserving raw evidence separately from model-visible evidence and classifying reconstruction fidelity. Map native memory identities into the target state and its request-local IDs.
3. Extend the small Python Observer loop only after inspecting the pilot. Add stateful replay, Reflector/Dropper handlers, and evaluation as needed. Keep Hatfield-only training and report native Pi transfer evaluation separately on held-out families.
4. Before remote labeling, check provider permissions and the data that may leave the machine. Run the selected teacher with the same input and contract as the student. Validate actions through isolated local handlers and review labels before exporting training examples. Local unredacted collection does not authorize sending secrets to a provider.

The design remains in [`hatfield-om-dataset-spec/OM-DATA-01.md`](hatfield-om-dataset-spec/OM-DATA-01.md). Model compatibility, token counts, family counts, and training readiness are not measured by this collector.
