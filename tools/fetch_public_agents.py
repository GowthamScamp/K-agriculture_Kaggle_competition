"""Read public Kaggle notebooks without credentials or executing notebook code.

Downloads are pinned by content hash. Agent extraction accepts only literal
``%%writefile main.py`` cells or literal Python source assignments; arbitrary
notebook builders are deliberately not executed. Use --inspect to investigate
an unsupported builder before adding a narrowly scoped extractor.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import tarfile
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {
    "market_smart": "tetsutani/market-smart-farming-kaggriculture",
    "recovering_harvests": "ahmedberatozer/kaggriculture-v43-recovering-lost-harvests",
    "rank_your_agent": "raykkretzschmar/kaggriculture-rank-your-agent",
}


def pull(ref: str) -> dict:
    request = urllib.request.Request(
        "https://www.kaggle.com/api/v1/kernels/pull/" + ref,
        headers={"User-Agent": "Kaggriculture-offline-benchmark/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def cells(payload: dict) -> list[str]:
    blob = payload["blob"]
    if isinstance(blob, dict):
        blob = blob.get("source", blob.get("sourceNullable", blob))
    notebook = json.loads(blob) if isinstance(blob, str) else blob
    if isinstance(notebook, dict) and "cells" in notebook:
        return ["".join(c.get("source", [])) for c in notebook["cells"]
                if c.get("cell_type") == "code"]
    return [str(blob)]


def literal_agent(code_cells: list[str]) -> str:
    candidates: list[str] = []
    literals: dict[str, str] = {}
    for code in code_cells:
        lines = code.splitlines(keepends=True)
        if lines and lines[0].strip() in ("%%writefile main.py", "%%writefile /kaggle/working/main.py"):
            candidates.append("".join(lines[1:]))
            continue
        if code.lstrip().startswith(("!", "%")):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                value = node.value.value
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(value, str):
                        literals[target.id] = value
                if isinstance(value, str) and "def agent(" in value:
                    candidates.append(value)
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SOURCE_BYTES" for t in node.targets)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Constant)
                    and node.value.func.value.value == b""
                    and node.value.func.attr == "join"):
                # V43's immutable bytes are a tuple of bytes literals, not code.
                parts = ast.literal_eval(node.value.args[0])
                if not isinstance(parts, tuple) or not all(isinstance(p, bytes) for p in parts):
                    raise ValueError("SOURCE_BYTES is not a literal bytes tuple")
                data = b"".join(parts)
                expected = literals.get("EXPECTED_MAIN_SHA256")
                if expected is None or hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError("V43 literal source hash does not match")
                return data.decode("utf-8")
    if "ARCHIVE_B64" in literals:
        archive = base64.b64decode(literals["ARCHIVE_B64"])
        if hashlib.sha256(archive).hexdigest() != literals["EXPECTED_ARCHIVE_SHA256"]:
            raise ValueError("Public notebook's archive hash does not match")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            data = bundle.extractfile("main.py").read()
        if hashlib.sha256(data).hexdigest() != literals["EXPECTED_MAIN_SHA256"]:
            raise ValueError("Public notebook's main.py hash does not match")
        return data.decode("utf-8")
    for name in ("SUBMISSION_B85", "AGENT_B85", "SOURCE_B85"):
        if name in literals:
            return zlib.decompress(base64.b85decode(literals[name].encode("ascii"))).decode("utf-8")
    if not candidates:
        raise ValueError("No literal agent; inspect the builder rather than executing it")
    return max(candidates, key=len)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--cached", action="store_true", help="use the pinned local notebook")
    parser.add_argument("--label", choices=sorted(PUBLIC), action="append")
    args = parser.parse_args()
    failed = False
    for label in args.label or PUBLIC:
        directory = ROOT / "benchmarks" / "public" / label
        directory.mkdir(parents=True, exist_ok=True)
        payload = (json.loads((directory / "notebook.json").read_bytes())
                   if args.cached else pull(PUBLIC[label]))
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        metadata = payload["metadata"]
        provenance = {
            "ref": PUBLIC[label], "url": "https://www.kaggle.com/code/" + PUBLIC[label],
            "metadata": metadata, "notebook_sha256": hashlib.sha256(raw).hexdigest(),
        }
        # Keep the full public notebook so extraction can be reproduced offline.
        (directory / "notebook.json").write_bytes(raw)
        code_cells = cells(payload)
        print(label, json.dumps(metadata), flush=True)
        if args.inspect:
            for index, code in enumerate(code_cells):
                print(f"  cell {index}, {len(code)} chars: " + json.dumps(code[:180]) + " ... " + json.dumps(code[-550:]))
        else:
            try:
                source = literal_agent(code_cells)
                if "Apache" not in source:
                    raise ValueError("Extracted agent lacks an Apache notice; do not reuse")
                ast.parse(source)
                data = source.encode("utf-8")
                (directory / "main.py").write_bytes(data)
                provenance["agent_sha256"] = hashlib.sha256(data).hexdigest()
                print(f"  extracted {len(data)} bytes, SHA-256 {provenance['agent_sha256']}", flush=True)
            except ValueError as error:
                provenance["extraction_error"] = str(error)
                failed = True
                print(f"  BLOCKED: {error}", flush=True)
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
