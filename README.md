# vecdex

[![ci](https://github.com/KlGamers/vecdex/actions/workflows/ci.yml/badge.svg)](https://github.com/KlGamers/vecdex/actions/workflows/ci.yml)

Índice de vecinos más cercanos aproximados (HNSW) escrito desde cero en C++17,
con bindings para Python. Sin dependencias en el núcleo: una cabecera y un
archivo fuente.

## Para qué sirve

La búsqueda semántica, los sistemas de recomendación y el RAG necesitan
encontrar los vectores más parecidos a uno dado entre cientos de miles o
millones. Compararlos todos es exacto pero lento. HNSW construye un grafo
navegable por capas que encuentra casi los mismos vecinos visitando una
fracción mínima de los vectores.

Este proyecto implementa el algoritmo del artículo de Malkov y Yashunin (2018)
completo: inserción incremental, selección heurística de vecinos, búsqueda con
`ef` ajustable y persistencia en disco.

## Resultados

Fashion-MNIST del conjunto [ann-benchmarks](https://github.com/erikbern/ann-benchmarks):
60,000 vectores de 784 dimensiones, 10,000 consultas, recall@10 contra los
vecinos exactos. Mismos parámetros para las dos librerías (`M=16`,
`ef_build=200`), un solo hilo, en una laptop Apple Silicon.

| ef | recall vecdex | QPS vecdex | recall hnswlib | QPS hnswlib | razón de QPS |
|---|---|---|---|---|---|
| 10 | 0.9328 | 4,461 | 0.9295 | 4,327 | 1.03 |
| 20 | 0.9792 | 3,301 | 0.9775 | 3,298 | 1.00 |
| 40 | 0.9945 | 1,994 | 0.9935 | 2,084 | 0.96 |
| 80 | 0.9983 | 1,439 | 0.9980 | 1,433 | 1.00 |
| 160 | 0.9993 | 912 | 0.9993 | 923 | 0.99 |
| 320 | 0.9997 | 543 | 0.9997 | 592 | 0.92 |

- Construcción del índice: vecdex 82.7 s, hnswlib 82.1 s.
- Búsqueda exacta con NumPy sobre los mismos datos: 151 consultas por segundo.
- vecdex con `ef=80` usando todos los núcleos: 9,328 consultas por segundo.

Qué dicen los números:

- vecdex obtiene el mismo recall que hnswlib y entre 0.92 y 1.03 veces su
  velocidad en esta máquina.
- Con `ef=80` responde 9.5 veces más rápido que la búsqueda exacta y devuelve
  el 99.8% de los vecinos correctos.
- La comparación es en ARM. hnswlib trae rutas SIMD escritas a mano solo para
  x86, así que en un procesador Intel o AMD debería sacar ventaja. No lo medí.

Para reproducirlo:

```sh
pip install ".[bench]"
python scripts/bench_ann.py
```

El script descarga el conjunto de datos (227 MB) en `data/`, que no se sube al
repositorio.

## Uso desde Python

```sh
pip install git+https://github.com/KlGamers/vecdex
```

```python
import numpy as np
import vecdex

vecs = np.random.rand(10_000, 128).astype(np.float32)

idx = vecdex.Index(dim=128, metric="cosine", m=16, ef_build=200)
idx.add(vecs)

ids, dists = idx.search(vecs[:5], k=10, ef=64, threads=0)

idx.save("demo.idx")
idx = vecdex.Index.load("demo.idx")
```

| Parámetro | Efecto |
|---|---|
| `metric` | `"l2"`, `"ip"` o `"cosine"` |
| `m` | Enlaces por nodo. Más enlaces dan mejor recall y usan más memoria |
| `ef_build` | Candidatos evaluados al insertar. Sube la calidad del grafo y el tiempo de construcción |
| `ef` | Candidatos evaluados al buscar. Es el control de recall contra velocidad |
| `threads` | Hilos para un lote de consultas. `0` usa todos los núcleos |

`add` acepta `ids` propios. Si faltan resultados, `search` rellena con `-1` y
distancia infinita.

## Uso desde C++

```cpp
#include "vecdex/index.hpp"

vecdex::Index idx(128, vecdex::Metric::L2);
idx.add(vec, 42);
for (const vecdex::Hit& h : idx.search(qry, 10, 64)) {
  std::printf("%llu %f\n", h.id, h.dist);
}
```

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Cómo está hecho

- **Memoria contigua.** Los vectores viven en un solo arreglo y los enlaces de
  la capa 0 en otro, con tamaño fijo por nodo. Recorrer vecinos no persigue
  punteros.
- **Capas superiores aparte.** Solo una fracción de los nodos sube de nivel,
  así que sus enlaces se guardan por nodo y no desperdician espacio.
- **Visitados por época.** Cada hilo marca nodos visitados con un contador en
  lugar de limpiar un arreglo en cada búsqueda.
- **Selección heurística.** Un candidato se conecta solo si está más cerca del
  nodo nuevo que de los vecinos ya elegidos. Evita que todos los enlaces
  apunten al mismo cúmulo.
- **Búsqueda concurrente.** `search` es `const` y no comparte estado mutable.
  El binding libera el GIL y reparte el lote entre hilos.
- **Carga defensiva.** `load` valida el tamaño de cada sección y cada enlace
  antes de aceptar el archivo. Un archivo truncado o corrupto lanza una
  excepción en lugar de leer fuera de memoria.

## Pruebas

- C++: recall contra fuerza bruta en las tres métricas, resultados exactos en
  conjuntos pequeños, determinismo, inserción incremental, ida y vuelta a
  disco y archivos corruptos. En CI corren con AddressSanitizer y UBSan.
- Python: formas y tipos, ids propios, relleno, igualdad entre uno y varios
  hilos, llamadas concurrentes y validación de entradas.

## Límites

- La construcción usa un solo hilo.
- No hay borrado de vectores.
- Hasta 2^32 vectores por índice.
- El archivo guardado usa el orden de bytes de la máquina que lo escribió.

## Licencia

MIT
