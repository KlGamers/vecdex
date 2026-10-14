# fashion-mnist-784-euclidean

- 60,000 vectors, 784 dimensions, 10,000 queries, recall@10
- M=16, ef_build=200, one thread for build and search
- machine: arm64, Darwin, Python 3.12.15
- build: vecdex 37.4 s, hnswlib 0.8.0 82.7 s
- exact search with NumPy: 151 QPS
- vecdex at ef=80 on all cores: 12,091 QPS

| ef | vecdex recall | vecdex QPS | hnswlib recall | hnswlib QPS | QPS ratio |
|---|---|---|---|---|---|
| 10 | 0.9328 | 10,412 | 0.9295 | 4,534 | 2.30 |
| 20 | 0.9792 | 6,043 | 0.9775 | 3,295 | 1.83 |
| 40 | 0.9945 | 4,293 | 0.9935 | 2,160 | 1.99 |
| 80 | 0.9983 | 2,618 | 0.9980 | 1,428 | 1.83 |
| 160 | 0.9993 | 1,623 | 0.9993 | 915 | 1.77 |
| 320 | 0.9997 | 1,003 | 0.9997 | 585 | 1.72 |
