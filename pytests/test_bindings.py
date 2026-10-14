import threading

import numpy as np
import pytest

from vecdex import Index


def data(n=3000, dim=24, seed=0):
    rng = np.random.default_rng(seed)
    cen = rng.normal(0, 4, (20, dim))
    return (cen[rng.integers(0, 20, n)] + rng.normal(0, 1, (n, dim))).astype(np.float32)


def brute(base, qrys, k):
    d = ((qrys[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


def recall(got, want):
    return np.mean([len(set(g) & set(w)) / len(w) for g, w in zip(got, want, strict=True)])


def test_recall_and_shapes():
    base, qrys = data(), data(100, seed=1)
    idx = Index(24)
    idx.add(base)
    assert len(idx) == 3000 and idx.dim == 24 and idx.metric == "l2" and idx.m == 16
    ids, dst = idx.search(qrys, k=10, ef=100)
    assert ids.shape == (100, 10) and dst.shape == (100, 10)
    assert ids.dtype == np.int64 and dst.dtype == np.float32
    assert (np.diff(dst, axis=1) >= 0).all()
    assert recall(ids, brute(base, qrys, 10)) >= 0.98
    one_id, one_d = idx.search(qrys[0], k=5)
    assert one_id.shape == (5,) and one_d.shape == (5,)


def test_custom_ids_and_padding():
    base = data(8, dim=4)
    idx = Index(4)
    idx.add(base, ids=np.arange(100, 108))
    ids, dst = idx.search(base[:2], k=12, ef=50)
    assert ids[0, 0] == 100 and ids[1, 0] == 101
    assert (ids[:, 8:] == -1).all() and np.isinf(dst[:, 8:]).all()
    assert sorted(ids[0, :8]) == list(range(100, 108))


def test_threads_match_single():
    base, qrys = data(), data(500, seed=2)
    idx = Index(24, metric="cosine")
    idx.add(base)
    a = idx.search(qrys, k=10, ef=64, threads=1)
    b = idx.search(qrys, k=10, ef=64, threads=4)
    c = idx.search(qrys, k=10, ef=64, threads=0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
    assert np.array_equal(a[1], b[1])


def test_concurrent_python_callers():
    base, qrys = data(), data(200, seed=3)
    idx = Index(24)
    idx.add(base)
    want = idx.search(qrys, k=5, ef=50)[0]
    out = [None] * 6

    def work(i):
        out[i] = idx.search(qrys, k=5, ef=50)[0]

    ths = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert all(np.array_equal(o, want) for o in out)


def test_save_load(tmp_path):
    base = data(1500)
    idx = Index(24, metric="ip", m=12, ef_build=80)
    idx.add(base)
    path = str(tmp_path / "a.idx")
    idx.save(path)
    back = Index.load(path)
    assert (len(back), back.metric, back.m, back.ef_build) == (1500, "ip", 12, 80)
    assert np.array_equal(idx.search(base[:50], 10, 60)[0], back.search(base[:50], 10, 60)[0])
    (tmp_path / "bad.idx").write_bytes(b"nope")
    with pytest.raises(RuntimeError):
        Index.load(str(tmp_path / "bad.idx"))


def test_input_validation():
    idx = Index(4)
    with pytest.raises(ValueError):
        idx.add(np.zeros((3, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        idx.add(np.zeros((3, 4), dtype=np.float32), ids=np.arange(2))
    with pytest.raises(ValueError):
        idx.search(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        Index(4, metric="hamming")
    with pytest.raises(ValueError):
        Index(0)
    idx.add(np.ones((2, 4), dtype=np.float64))
    assert len(idx) == 2
    ids, _ = Index(4).search(np.zeros(4, dtype=np.float32), k=3)
    assert (ids == -1).all()
