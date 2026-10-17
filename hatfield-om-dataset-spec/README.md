# OM-DATA-01 specification package

Start with **OM-DATA-01.md**. This is a proposed dataset-generation and evaluation design,
not an implemented exporter, an actual training dataset, or a measured model comparison.

## Training plans

- [OM-PLAN-02: First joint memory-writer experiment](OM-PLAN-02-judge-guided-training.md): export captured traces, prove student compatibility, review one bounded batch automatically, train all three roles with one adapter, and evaluate on held-out history. Further Jev calibration and manual curation are not prerequisites.
- [OM-PLAN-01: Model training pipeline](OM-PLAN-01-model-training-pipeline.md): broader project outline. Follow OM-PLAN-02 where it changes the work order or candidate-selection policy.

OM-DATA-01 retains the provenance, privacy, runtime-contract, and evaluation requirements. The plans do not establish that training prerequisites have passed.

## Judge pilot

- [Jev quality rubric](om-jev-quality-rubric.md): useful coverage, context recoverability, observation importance, compression, and safe removal, each rated pass, needs work, or uncertain when applicable.
- [Synthetic controls](fixtures/judge-controls.json): false completion, invented reflection, unsafe removal, useful compression, and overcompression that loses a constraint. These are not real session data or approved training labels.
- [Additional Jev controls](fixtures/jev-quality-controls.json): required identifiers and observation relevance ratings.
- [Reflector and Dropper controls](fixtures/jev-reflector-dropper-controls.json): preserved critical facts, last-copy deletion, current-state revisions, and valid selective no-ops. These test Jev's judging prompts, not changed worker prompts.
- [Run the Jev judge](../README.md#try-the-jev-judge): native typed quality checks, record or source locations, and pairwise preferences on blinded cases.
- [Earlier DeepSeek prompt](om-judge-prompt.md): the earlier issue taxonomy and prompted JSON response contract.
- [Earlier DeepSeek judge](../README.md#try-the-prompted-judge): high-reasoning comparison trial using prompted JSON and exact quotation checks.

## Structural fixture

- `example.schema.json`: draft structural schema for canonical invocation records.
- `schemas/*.arguments.schema.json`: task action argument shapes, not byte-identical production tool schemas.
- `fixtures/observer-example.json`: synthetic positive Observer trace with two facts and a receipt.
- `fixtures/blobs/`: complete fixture inputs, states, tool definitions, and audit facts.
- `validate_fixture.py`: structural/hash/basic trace validation only.

The fixture policy is intentionally small. Its state IDs use a fixture-only hash construction.
It must not be substituted for Hatfield's prompts, production ID generation, or real handler execution.
No real session or personal data is included. Token counts and actual model execution are not claimed.

Run validation with Python 3.10+ and `jsonschema` installed:

```bash
python validate_fixture.py
```

All referenced fixture files use package-relative paths. Example identity is SHA-256 of
canonical JSON of the complete record excluding `example_id`; the checked-input hash covers
canonical JSON of `input`. This fixture's canonical JSON uses sorted keys, compact separators,
UTF-8, and unescaped Unicode. Production releases must freeze equivalent cross-language
canonicalization and validate their own full runtime/semantic requirements.
