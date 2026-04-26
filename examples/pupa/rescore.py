#!/usr/bin/env python3
"""Rescore the best bundle from a snapshot on the PUPA test set (221 examples)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from problem import load_splits, score_fn
else:
    from .problem import load_splits, score_fn


def main() -> None:
    snapshot_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "runs" / "20260425_140643" / "snapshot.json"
    )
    print(f"[Rescore] Loading snapshot: {snapshot_path}")
    snap = json.loads(snapshot_path.read_text())
    elites = snap["elites"]
    best = max(elites, key=lambda e: e.get("primary_score") or 0)
    bundle = json.loads(best["content"])["prompts"]
    print(f"[Rescore] Best proxy score: {best.get('primary_score'):.4f}")
    print(f"[Rescore] Bundle keys: {list(bundle.keys())}")

    _train, testset = load_splits()
    print(f"[Rescore] Test set size: {len(testset)}")

    t0 = time.time()
    result = score_fn(bundle, testset)
    elapsed = time.time() - t0

    print(f"[Rescore] Done in {elapsed:.1f}s")
    print(
        f"[Rescore] Test score = {result['score']:.4f} | "
        f"quality_mean = {result['quality_mean']:.4f} | "
        f"leakage_mean = {result['leakage_mean']:.4f} | "
        f"failures = {result['request_failures']}"
    )

    out_path = snapshot_path.parent / "test_result.json"
    out_path.write_text(
        json.dumps(
            {
                "best_bundle": bundle,
                "proxy_score": best.get("primary_score"),
                "test_score": result["score"],
                "quality_mean": result["quality_mean"],
                "leakage_mean": result["leakage_mean"],
                "request_failures": result["request_failures"],
                "n_test": len(testset),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )
    print(f"[Rescore] Wrote {out_path}")


if __name__ == "__main__":
    main()
