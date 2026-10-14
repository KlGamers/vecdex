from __future__ import annotations

import argparse
import platform
import shutil
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

import h5py
import hnswlib
import numpy as np

import vecdex

SRC = "http://ann-benchmarks.com/"


def fetch(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"{name}.hdf5"
    if not dst.exists():
        req = urllib.request.Request(f"{SRC}{name}.hdf5", headers={"User-Agent": "vecdex-bench"})
        with urllib.request.urlopen(req) as res, open(dst, "wb") as fh:
            shutil.copyfileobj(res, fh)
    return dst


def recall(got: np.ndarray, want: np.ndarray, k: int) -> float:
    hit = sum(len(set(g[:k]) & set(w[:k])) for g, w in zip(got, want, strict=True))
    return hit / (len(want) * k)


def timed(fn, reps: int = 3) -> tuple[float, np.ndarray]:
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    par = argparse.ArgumentParser()
    par.add_argument("--set", default="fashion-mnist-784-euclidean")
    par.add_argument("--data", default="data")
    par.add_argument("--out", default="bench/RESULTS.md")
    par.add_argument("--m", type=int, default=16)
    par.add_argument("--ef-build", type=int, default=200)
    par.add_argument("--k", type=int, default=10)
    args = par.parse_args()

    with h5py.File(fetch(Path(args.data), args.set)) as fh:
        base = np.ascontiguousarray(fh["train"][:], dtype=np.float32)
        qrys = np.ascontiguousarray(fh["test"][:], dtype=np.float32)
        want = fh["neighbors"][:]
    n, dim = base.shape
    k = args.k

    t0 = time.perf_counter()
    own = vecdex.Index(dim, "l2", args.m, args.ef_build)
    own.add(base)
    own_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    ref = hnswlib.Index("l2", dim)
    ref.set_num_threads(1)
    ref.init_index(n, M=args.m, ef_construction=args.ef_build, random_seed=42)
    ref.add_items(base, num_threads=1)
    ref_build = time.perf_counter() - t0

    sub = qrys[:200]
    sq = (base**2).sum(1)
    t0 = time.perf_counter()
    exact = np.argsort(sq[None, :] - 2 * sub @ base.T, axis=1)[:, :k]
    brute_qps = len(sub) / (time.perf_counter() - t0)
    assert recall(exact, want[:200], k) > 0.999

    rows = [
        "| ef | vecdex recall | vecdex QPS | hnswlib recall | hnswlib QPS | QPS ratio |",
        "|---|---|---|---|---|---|",
    ]
    for ef in (10, 20, 40, 80, 160, 320):
        sec, got = timed(lambda ef=ef: own.search(qrys, k, ef, threads=1)[0])
        ref.set_ef(ef)
        rsec, rgot = timed(lambda: ref.knn_query(qrys, k, num_threads=1)[0])
        own_qps, ref_qps = len(qrys) / sec, len(qrys) / rsec
        rows.append(
            f"| {ef} | {recall(got, want, k):.4f} | {own_qps:,.0f} | {recall(rgot, want, k):.4f} "
            f"| {ref_qps:,.0f} | {own_qps / ref_qps:.2f} |"
        )
        print(rows[-1], flush=True)

    sec, _ = timed(lambda: own.search(qrys, k, 80, threads=0))
    head = [
        f"# {args.set}",
        "",
        f"- {n:,} vectors, {dim} dimensions, {len(qrys):,} queries, recall@{k}",
        f"- M={args.m}, ef_build={args.ef_build}, one thread for build and search",
        f"- machine: {platform.machine()}, {platform.system()}, Python {platform.python_version()}",
        f"- build: vecdex {own_build:.1f} s, hnswlib {version('hnswlib')} {ref_build:.1f} s",
        f"- exact search with NumPy: {brute_qps:,.0f} QPS",
        f"- vecdex at ef=80 on all cores: {len(qrys) / sec:,.0f} QPS",
        "",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(head + rows) + "\n")
    print("\n".join(head))


if __name__ == "__main__":
    main()
