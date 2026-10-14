#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vecdex/index.hpp"

namespace py = pybind11;
using vecdex::Hit;
using vecdex::Index;
using vecdex::Metric;

namespace {

using FArr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

Metric parse(const std::string& name) {
  if (name == "l2") return Metric::L2;
  if (name == "ip") return Metric::IP;
  if (name == "cosine") return Metric::Cosine;
  throw py::value_error("metric must be 'l2', 'ip' or 'cosine'");
}

const char* label(Metric mt) {
  switch (mt) {
    case Metric::L2:
      return "l2";
    case Metric::IP:
      return "ip";
    default:
      return "cosine";
  }
}

std::size_t rows(const FArr& arr, std::size_t dim) {
  if (arr.ndim() == 1 && static_cast<std::size_t>(arr.shape(0)) == dim) return 1;
  if (arr.ndim() == 2 && static_cast<std::size_t>(arr.shape(1)) == dim) {
    return static_cast<std::size_t>(arr.shape(0));
  }
  throw py::value_error("expected an array of shape (n, " + std::to_string(dim) + ")");
}

void add(Index& idx, const FArr& vecs, const std::optional<IArr>& ids) {
  const std::size_t n = rows(vecs, idx.dim());
  if (ids && (ids->ndim() != 1 || static_cast<std::size_t>(ids->shape(0)) != n)) {
    throw py::value_error("ids must be a 1-d array with one entry per vector");
  }
  const float* src = vecs.data();
  const std::int64_t* tag = ids ? ids->data() : nullptr;
  const std::size_t base = idx.size();
  py::gil_scoped_release free;
  idx.reserve(base + n);
  for (std::size_t i = 0; i < n; ++i) {
    idx.add(src + i * idx.dim(), tag ? static_cast<std::uint64_t>(tag[i]) : base + i);
  }
}

py::tuple search(const Index& idx, const FArr& qrys, std::size_t k, std::size_t ef, int threads) {
  const std::size_t n = rows(qrys, idx.dim());
  const bool flat = qrys.ndim() == 1;
  py::array_t<std::int64_t> ids({n, k});
  py::array_t<float> dst({n, k});
  std::int64_t* out_id = ids.mutable_data();
  float* out_d = dst.mutable_data();
  const float* src = qrys.data();
  {
    py::gil_scoped_release free;
    std::atomic<std::size_t> next{0};
    auto work = [&] {
      for (std::size_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
        const std::vector<Hit> hits = idx.search(src + i * idx.dim(), k, ef);
        for (std::size_t j = 0; j < k; ++j) {
          const bool has = j < hits.size();
          out_id[i * k + j] = has ? static_cast<std::int64_t>(hits[j].id) : -1;
          out_d[i * k + j] = has ? hits[j].dist : std::numeric_limits<float>::infinity();
        }
      }
    };
    std::size_t cnt = threads > 0 ? static_cast<std::size_t>(threads) : std::thread::hardware_concurrency();
    cnt = std::max<std::size_t>(1, std::min(cnt, n));
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < cnt; ++t) pool.emplace_back(work);
    work();
    for (std::thread& th : pool) th.join();
  }
  if (flat) return py::make_tuple(ids.reshape({k}), dst.reshape({k}));
  return py::make_tuple(ids, dst);
}

}

PYBIND11_MODULE(_vecdex, mod) {
  py::class_<Index>(mod, "Index")
      .def(py::init([](std::size_t dim, const std::string& metric, std::size_t m, std::size_t ef_build,
                       std::uint64_t seed) { return Index(dim, parse(metric), m, ef_build, seed); }),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("m") = 16, py::arg("ef_build") = 200,
           py::arg("seed") = 42)
      .def("add", &add, py::arg("vecs"), py::arg("ids") = std::nullopt)
      .def("search", &search, py::arg("qrys"), py::arg("k") = 10, py::arg("ef") = 0,
           py::arg("threads") = 1)
      .def("save", &Index::save, py::arg("path"))
      .def_static("load", &Index::load, py::arg("path"))
      .def("__len__", &Index::size)
      .def_property_readonly("dim", &Index::dim)
      .def_property_readonly("m", &Index::m)
      .def_property_readonly("ef_build", &Index::ef_build)
      .def_property_readonly("max_level", &Index::max_level)
      .def_property_readonly("metric", [](const Index& idx) { return std::string(label(idx.metric())); });
}
