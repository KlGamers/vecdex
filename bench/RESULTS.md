# fashion-mnist-784-euclidean

- 60,000 vectors, 784 dimensions, 10,000 queries, recall@10
- M=16, ef_build=200, one thread for build and search
- machine: arm64, Darwin, Python 3.12.15
- build: vecdex 82.7 s, hnswlib 0.8.0 82.1 s
- exact search with NumPy: 151 QPS
- vecdex at ef=80 on all cores: 9,328 QPS

| ef | vecdex recall | vecdex QPS | hnswlib recall | hnswlib QPS | QPS ratio |
|---|---|---|---|---|---|
| 10 | 0.9328 | 4,461 | 0.9295 | 4,327 | 1.03 |
| 20 | 0.9792 | 3,301 | 0.9775 | 3,298 | 1.00 |
| 40 | 0.9945 | 1,994 | 0.9935 | 2,084 | 0.96 |
| 80 | 0.9983 | 1,439 | 0.9980 | 1,433 | 1.00 |
| 160 | 0.9993 | 912 | 0.9993 | 923 | 0.99 |
| 320 | 0.9997 | 543 | 0.9997 | 592 | 0.92 |
