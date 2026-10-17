"""Redact model-visible credentials without changing character offsets or source IDs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

CATALOG_PATH = Path(".hatfield/security/redaction-values.json")
MARKER = "[REDACTED]"
TOKEN = re.compile(
    r"(?<![\w])(?:sk-[A-Za-z0-9_-]{20,}|(?:gh[opusr]_|github_pat_|npm_|sgp_|rpa_|ctx7sk-)[A-Za-z0-9_-]{20,}"
    r"|eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,})"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----", re.S)
SENSITIVE_NAME = r"[A-Za-z0-9_]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|password|passwd|client[_-]?secret|secret[_-]?key|private[_-]?key)"
ASSIGNMENT = re.compile(
    r"(?i)(?P<prefix>\b" + SENSITIVE_NAME + r"(?:\\*[\"'])?\]?\s*(?:=>|[:=])\s*)(?:\\*[\"'](?P<quoted>[^\"'\r\n]*?)\\*[\"']|(?P<bare>[A-Za-z0-9._~+/=-]+))"
)
AUTH = re.compile(r"(?i)\b(?:Bearer|Basic)\s+(?P<value>[A-Za-z0-9._~+/=-]{12,})")
URL_PASSWORD = re.compile(r"(?i)\b(?:https?|postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?)://[^/\s:@]+:(?P<value>[^@\s]+)@")
IDENTITIES = {"run_id", "session_id", "candidate_id", "native_entry_id", "native_memory_id", "source_refs",
              "source_archive_ref", "state_scope", "state_parent_id", "state_before_id", "historical_range_id",
              "historical_source_digest", "historical_part_digest", "native_identity"}


def _environment_values():
    return {v for k, v in os.environ.items() if len(v) >= 12 and
            (k.endswith(("_API_KEY", "_TOKEN", "_SECRET")) or k == "ANTHROPIC_AUTH_TOKEN")}


def _mask(value):
    if value.startswith(MARKER) or not value.strip("*\r\n"):
        return value
    masked = re.sub(r"[^\r\n]", "*", value)
    return MARKER + masked[len(MARKER):] if len(masked.splitlines()[0]) >= len(MARKER) else masked


def _placeholder(value):
    return value.startswith(("$", "os.environ", MARKER)) or value.endswith(("_API_KEY", "_TOKEN"))


class Redactor:
    def __init__(self, values=(), catalog=CATALOG_PATH):
        self.catalog_loaded = catalog is not None and Path(catalog).is_file()
        saved = json.loads(Path(catalog).read_text())["values"] if self.catalog_loaded else []
        policy_values = set(values) | set(saved)
        matches = sorted(policy_values | _environment_values(), key=lambda v: (-len(v), v))
        self._known = re.compile("|".join(re.escape(v) for v in matches)) if matches else None
        identity = {"implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "values": sorted(hashlib.sha256(v.encode()).hexdigest() for v in policy_values)}
        # Rotating transport credentials must not invalidate completed replay jobs.
        # Environment values remain matchers; changed visible text still changes the job.
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def text(self, text):
        if self._known is not None:
            text = self._known.sub(lambda m: _mask(m.group()), text)
        text = TOKEN.sub(lambda m: _mask(m.group()), text)
        text = PRIVATE_KEY.sub(lambda m: _mask(m.group()), text)

        def assignment(match):
            group = "quoted" if match.group("quoted") is not None else "bare"
            value = match.group(group)
            if _placeholder(value):
                return match.group()
            start, end = match.span(group)
            start, end = start - match.start(), end - match.start()
            return match.group()[:start] + _mask(value) + match.group()[end:]

        text = ASSIGNMENT.sub(assignment, text)
        def credential(match):
            start, end = match.start("value") - match.start(), match.end("value") - match.start()
            return match.group()[:start] + _mask(match.group("value")) + match.group()[end:]

        return URL_PASSWORD.sub(credential, AUTH.sub(credential, text))

    def tree(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.tree(v) for v in value]
        if isinstance(value, dict):
            clean = {}
            for key, item in value.items():
                if self.text(key) != key:
                    raise ValueError("Credential used as an object key; refusing to rewrite it")
                if isinstance(item, str) and re.fullmatch(SENSITIVE_NAME, key, re.I) and not _placeholder(item):
                    clean[key] = _mask(item)
                else:
                    clean[key] = self.tree(item)
            return clean
        return value

    def job(self, job):
        clean = self.tree(job)

        def check(before, after):
            if isinstance(before, dict):
                for key, value in before.items():
                    if key in IDENTITIES and value != after[key]:
                        raise ValueError("Credential detected in a provenance identity; refusing to rewrite it")
                    check(value, after[key])
            elif isinstance(before, list):
                for a, b in zip(before, after):
                    check(a, b)

        check(job, clean)
        clean["redaction_id"] = self.fingerprint
        return clean

    def require_clean(self, value):
        if self.tree(value) != value:
            raise ValueError("Unredacted credential detected; refusing to send or reuse this record")


def build_catalog(report=Path("data/security-scan/gitguardian-archives.json"), output=CATALOG_PATH):
    """Recover detected values locally; never put them in logs or training artifacts."""
    values = _environment_values()
    occurrences = 0
    for entity in json.loads(report.read_text())["entities_with_incidents"]:
        if not entity["incidents"]:
            continue
        lines = Path(entity["filename"]).read_text().splitlines(keepends=True)
        for incident in entity["incidents"]:
            for occurrence in incident["occurrences"]:
                if occurrence["line_start"] != occurrence["line_end"]:
                    raise ValueError("Multiline scanner coordinates require manual review")
                line = lines[occurrence["line_start"] - 1]
                value = line[occurrence["index_start"]:occurrence["index_end"]]
                masked = occurrence["match"]
                if len(value) != len(masked) or any(a != b and b != "*" for a, b in zip(value, masked)):
                    raise ValueError("Scanner coordinates do not match the frozen source")
                occurrences += 1
                if len(value) >= 12 and value.strip("*") and not value.startswith(MARKER):
                    values.add(value)
                    try:
                        decoded = json.loads('"' + value + '"')
                        if len(decoded) >= 12:
                            values.add(decoded)
                    except ValueError:
                        pass
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as target:
        json.dump({"version": 1, "values": sorted(values)}, target, ensure_ascii=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    return {"catalog_values": len(values), "scanner_occurrences_reviewed": occurrences}
