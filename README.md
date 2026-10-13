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

1. Establish lineage groups and duplicate checks, then freeze train, validation, and test assignments before labeling.
2. Construct causal Hatfield-contract invocations from the frozen sources. Preserve raw evidence separately from model-visible evidence and classify reconstruction fidelity.
3. Before remote labeling, check provider permissions and the data that may leave the machine. Local unredacted collection does not authorize sending secrets to a provider.
4. Run the selected teacher, validate actions through isolated Hatfield handlers, and review labels before exporting training examples.

The design remains in [`hatfield-om-dataset-spec/OM-DATA-01.md`](hatfield-om-dataset-spec/OM-DATA-01.md). Model compatibility, token counts, family counts, and training readiness are not measured by this collector.
