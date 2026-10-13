# OM-DATA-01 specification package

Start with **OM-DATA-01.md**. This is a proposed dataset-generation and evaluation design,
not an implemented exporter, an actual training dataset, or a measured model comparison.

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
