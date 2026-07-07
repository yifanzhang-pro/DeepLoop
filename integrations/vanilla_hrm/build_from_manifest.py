#!/usr/bin/env python
"""Deterministic, hash-verified rebuild of HRM's ARC datasets on ANY machine.

Why this exists
---------------
Upstream `dataset/build_arc_dataset.py` enumerates puzzle files with
`os.scandir()` + `glob()` — both return entries in FILESYSTEM-DEPENDENT order —
and then feeds that unsorted list into a seeded `np.random.shuffle` plus
per-puzzle augmentation draws. The RNG stream is seeded, but its *input order*
is not, so the same script + same seed produces a different dataset on a
different machine (verified: two same-seed builds shared only 0.14% of their
augmentations). The published `sapientinc/HRM-checkpoint-ARC-2` has a
per-augmentation `puzzle_emb` welded to ONE specific build, so evaluating it on
any other build silently degrades K=2 from ~5% to ~1.7%.

What this does
--------------
1. Imports upstream `dataset/build_arc_dataset.py` UNMODIFIED.
2. Replaces only its filesystem enumeration with the order recorded in a
   manifest (captured from the reference machine), making the entire seeded
   RNG stream — and therefore the output — byte-reproducible everywhere.
   numpy's legacy `np.random.*` generator is version-frozen (NEP 19), so the
   manifest + seed pins everything.
3. md5-verifies every output file against reference_hashes.json and exits
   non-zero on any mismatch. No silent divergence possible.

Usage (from anywhere; ~120 GB RAM for arc2 — use a fat CPU node):
    python build_from_manifest.py \
        --hrm-dir /path/to/HRM \
        --dataset arc2 \
        --package-dir /path/to/this/dir   # default: this script's directory
        [--output-dir data/arc-2-aug-1000]  [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ManifestScandir:
    """os-module proxy: scandir() of a manifest-covered dir yields entries in
    manifest order; everything else passes through to the real os module."""

    class _Entry:
        def __init__(self, path: str):
            self.path = path
            self.name = os.path.basename(path)

        def is_dir(self) -> bool:
            return True

    def __init__(self, real_os, manifest: dict):
        self._os = real_os
        self._manifest = manifest

    def scandir(self, path: str):
        key = os.path.normpath(path)
        for ds, spec in self._manifest.items():
            if os.path.normpath(ds) == key:
                return iter(
                    self._Entry(os.path.join(path, sub)) for sub in spec["subdir_order"]
                )
        return self._os.scandir(path)

    def __getattr__(self, name):
        return getattr(self._os, name)


def make_manifest_glob(manifest: dict):
    def manifest_glob(pattern: str):
        d = os.path.normpath(os.path.dirname(pattern))
        for ds, spec in manifest.items():
            for sub, files in spec["files"].items():
                if os.path.normpath(os.path.join(ds, sub)) == d:
                    return [os.path.join(os.path.dirname(pattern), f) for f in files]
        raise RuntimeError(
            f"manifest has no entry for glob pattern '{pattern}' — refusing to "
            "fall back to filesystem order (that is the bug this script exists "
            "to prevent). Regenerate the manifest or check --dataset."
        )

    return manifest_glob


def main() -> int:
    # HARD determinism requirement: upstream build_arc_dataset.py iterates a
    # set of (split, subset) STRING TUPLES (`dests = set(dest_mapping.values())`,
    # line ~103), whose order depends on Python's per-process hash
    # randomization. That order decides whether the train or test split is
    # created (and therefore SAVED) first, which shifts where the train-split
    # translational-pad RNG draws sit in the seeded stream — a per-process
    # 50/50 coin yielding exactly two different-but-internally-deterministic
    # builds (identifiers and the test split are invariant; train/{inputs,
    # labels}.npy flip). Pin the hash seed and re-exec so every build takes
    # the same branch on every machine.
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = dict(os.environ, PYTHONHASHSEED="0")
        print("[determinism] re-executing with PYTHONHASHSEED=0")
        sys.stdout.flush()
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hrm-dir", required=True)
    ap.add_argument("--dataset", required=True, choices=["arc1", "arc2"])
    ap.add_argument("--package-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--output-dir", default=None,
                    help="default: the reference output_dir from reference_hashes.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-aug", type=int, default=1000)
    ap.add_argument("--skip-verify", action="store_true",
                    help="build only (e.g. for a deliberately different seed)")
    ap.add_argument("--skip-raw-check", action="store_true",
                    help="skip the raw-data submodule commit verification")
    args = ap.parse_args()

    manifest_path = os.path.join(args.package_dir, "manifests", f"{args.dataset}_enum_manifest.json")
    hashes_path = os.path.join(args.package_dir, "reference_hashes.json")
    manifest = json.load(open(manifest_path))
    all_ref = json.load(open(hashes_path))
    ref = all_ref[args.dataset]
    out_dir = args.output_dir or ref["output_dir"]

    # Run from HRM_DIR: upstream uses relative dataset_dirs/output_dir.
    os.chdir(args.hrm_dir)

    # Raw-data VERSION check: identical file *names* are not enough — content
    # matters. Real incident: HRM@ac15626 pins ConceptARC@b22ef52, but
    # upstream later fixed an output grid in corpus/Order/Order5.json
    # (0e67da6); the reference arc1 build uses the FIXED data, and a build
    # from the old pin silently differs in train/all__labels.npy only.
    if not args.skip_raw_check:
        import subprocess
        bad = []
        for sub, expect in all_ref.get("raw_data_commits", {}).items():
            sub_parts = os.path.normpath(sub).split(os.sep)
            used = any(os.path.normpath(ds).split(os.sep)[: len(sub_parts)] == sub_parts
                       for ds in manifest)
            if not used:
                continue  # submodule not used by this dataset
            try:
                head = subprocess.run(["git", "-C", sub, "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=True).stdout.strip()
                dirty = subprocess.run(["git", "-C", sub, "status", "--porcelain"],
                                       capture_output=True, text=True, check=True).stdout.strip()
            except Exception as e:  # not a git checkout (e.g. tarball deploy)
                print(f"[raw-check] WARNING: cannot git-inspect {sub} ({e}); skipping its check.")
                continue
            state = head + (" (DIRTY)" if dirty else "")
            ok = head == expect and not dirty
            print(f"[raw-check] {'OK  ' if ok else 'FAIL'} {sub}: {state}" + ("" if ok else f"  expected {expect}"))
            if not ok:
                bad.append((sub, expect))
        if bad:
            print("\nFATAL: raw-data submodule(s) differ from the reference build's version.")
            print("Fix (fetch by SHA works on GitHub even for unreferenced commits):")
            for sub, expect in bad:
                print(f"    git -C {sub} fetch origin {expect} && git -C {sub} checkout {expect}")
            print("Or pass --skip-raw-check to build anyway (output hashes will then likely FAIL).")
            return 3
    # build_arc_dataset.py uses sibling imports (`from common import ...`),
    # which resolve when it runs as a script (sys.path[0] = dataset/). Mirror
    # that exactly: put dataset/ on sys.path and import the module directly.
    sys.path.insert(0, os.path.join(args.hrm_dir, "dataset"))

    # Raw data presence check (manifest files must all exist).
    missing = [
        os.path.join(ds, sub, f)
        for ds, spec in manifest.items()
        for sub, files in spec["files"].items()
        for f in files
        if not os.path.isfile(os.path.join(ds, sub, f))
    ]
    if missing:
        print(f"FATAL: {len(missing)} raw files in the manifest are missing, e.g. {missing[:3]}")
        print("Run: git submodule update --init --recursive")
        return 1

    import build_arc_dataset as B  # noqa: E402  (upstream, unmodified)

    # Surgical enumeration override — only within the imported module.
    B.glob = make_manifest_glob(manifest)
    B.os = _ManifestScandir(os, manifest)

    cfg = B.DataProcessConfig(
        dataset_dirs=list(manifest.keys()),
        output_dir=out_dir,
        seed=args.seed,
        num_aug=args.num_aug,
    )
    print(f"[build] dataset={args.dataset} seed={args.seed} -> {out_dir} (manifest-pinned order)")
    B.convert_dataset(cfg)

    if args.skip_verify:
        print("[build] done (verification skipped by --skip-verify)")
        return 0

    print("[verify] md5 vs reference_hashes.json ...")
    bad = []
    for rel, expect in ref["files"].items():
        p = os.path.join(out_dir, rel)
        got = md5(p) if os.path.isfile(p) else "<missing>"
        status = "OK " if got == expect else "FAIL"
        print(f"  [{status}] {rel}  {got}")
        if got != expect:
            bad.append(rel)
    if bad:
        print(f"\nFATAL: {len(bad)} file(s) differ from the reference build: {bad}")
        print("This build is NOT comparable to the reference / published checkpoint.")
        print("Check: raw-data submodule commit, numpy installed, manifest integrity.")
        return 2
    print("\n[verify] ALL HASHES MATCH — build is byte-identical to the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
