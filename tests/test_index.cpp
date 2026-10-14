#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "vecdex/index.hpp"

using vecdex::Hit;
using vecdex::Index;
using vecdex::Metric;

namespace {

int g_fail = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
      ++g_fail;                                                            \
    }                                                                      \
  } while (0)

std::vector<float> clustered(std::size_t n, std::size_t dim, std::size_t groups, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> wide(0.0f, 4.0f), tight(0.0f, 1.0f);
  std::vector<float> cen(groups * dim);
  for (float& c : cen) c = wide(rng);
  std::vector<float> out(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t g = rng() % groups;
    for (std::size_t j = 0; j < dim; ++j) out[i * dim + j] = cen[g * dim + j] + tight(rng);
  }
  return out;
}

float truth(Metric mt, const float* a, const float* b, std::size_t dim) {
  double dp = 0, na = 0, nb = 0, sq = 0;
  for (std::size_t i = 0; i < dim; ++i) {
    dp += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
    sq += (a[i] - b[i]) * (a[i] - b[i]);
  }
  if (mt == Metric::L2) return static_cast<float>(sq);
  if (mt == Metric::IP) return static_cast<float>(1.0 - dp);
  return static_cast<float>(1.0 - dp / std::sqrt(na * nb));
}

std::vector<std::uint64_t> brute(Metric mt, const std::vector<float>& data, const float* qry,
                                 std::size_t dim, std::size_t k) {
  const std::size_t n = data.size() / dim;
  std::vector<std::pair<float, std::uint64_t>> all(n);
  for (std::size_t i = 0; i < n; ++i) all[i] = {truth(mt, data.data() + i * dim, qry, dim), i};
  std::partial_sort(all.begin(), all.begin() + static_cast<long>(k), all.end());
  std::vector<std::uint64_t> out(k);
  for (std::size_t i = 0; i < k; ++i) out[i] = all[i].second;
  return out;
}

double recall(const Index& idx, Metric mt, const std::vector<float>& data,
              const std::vector<float>& qrys, std::size_t dim, std::size_t k, std::size_t ef) {
  std::size_t hit = 0, tot = 0;
  for (std::size_t q = 0; q < qrys.size() / dim; ++q) {
    const float* qv = qrys.data() + q * dim;
    const auto want = brute(mt, data, qv, dim, k);
    const std::set<std::uint64_t> ws(want.begin(), want.end());
    for (const Hit& h : idx.search(qv, k, ef)) hit += ws.count(h.id);
    tot += k;
  }
  return static_cast<double>(hit) / static_cast<double>(tot);
}

Index build(Metric mt, const std::vector<float>& data, std::size_t dim) {
  Index idx(dim, mt, 16, 200);
  idx.reserve(data.size() / dim);
  for (std::size_t i = 0; i < data.size() / dim; ++i) idx.add(data.data() + i * dim, i);
  return idx;
}

void test_recall() {
  const std::size_t dim = 32;
  const auto data = clustered(6000, dim, 40, 1);
  const auto qrys = clustered(200, dim, 40, 1);
  for (Metric mt : {Metric::L2, Metric::Cosine, Metric::IP}) {
    auto pts = data;
    if (mt == Metric::IP) {
      for (std::size_t i = 0; i < pts.size() / dim; ++i) {
        double nrm = 0;
        for (std::size_t j = 0; j < dim; ++j) nrm += pts[i * dim + j] * pts[i * dim + j];
        for (std::size_t j = 0; j < dim; ++j) pts[i * dim + j] /= static_cast<float>(std::sqrt(nrm));
      }
    }
    const Index idx = build(mt, pts, dim);
    const double lo = recall(idx, mt, pts, qrys, dim, 10, 10);
    const double hi = recall(idx, mt, pts, qrys, dim, 10, 200);
    std::printf("  metric %u  recall@10  ef=10: %.3f  ef=200: %.3f\n", static_cast<unsigned>(mt), lo, hi);
    CHECK(hi >= 0.98);
    CHECK(hi >= lo);
  }
}

void test_exact_on_small_sets() {
  const std::size_t dim = 8;
  const auto data = clustered(50, dim, 5, 2);
  const Index idx = build(Metric::L2, data, dim);
  for (std::size_t i = 0; i < 50; ++i) {
    const auto got = idx.search(data.data() + i * dim, 1, 50);
    CHECK(got.size() == 1 && got[0].id == i && got[0].dist == 0.0f);
  }
  const auto all = idx.search(data.data(), 500, 500);
  CHECK(all.size() == 50);
  for (std::size_t i = 1; i < all.size(); ++i) CHECK(all[i - 1].dist <= all[i].dist);
  const auto want = brute(Metric::L2, data, data.data(), dim, 50);
  for (std::size_t i = 0; i < 50; ++i) CHECK(all[i].id == want[i] || all[i].dist == truth(Metric::L2, data.data() + want[i] * dim, data.data(), dim));
}

void test_edges() {
  Index idx(4);
  const float q[4] = {1, 2, 3, 4};
  CHECK(idx.search(q, 5).empty());
  idx.add(q, 77);
  CHECK(idx.search(q, 0).empty());
  auto one = idx.search(q, 5);
  CHECK(one.size() == 1 && one[0].id == 77);
  for (int i = 0; i < 100; ++i) idx.add(q, 1000 + static_cast<std::uint64_t>(i));
  auto dup = idx.search(q, 10, 64);
  CHECK(dup.size() == 10);
  for (const Hit& h : dup) CHECK(h.dist == 0.0f);

  Index cs(3, Metric::Cosine);
  const float a[3] = {10, 0, 0}, b[3] = {0, 5, 0}, c[3] = {0.1f, 0, 0}, zero[3] = {0, 0, 0};
  cs.add(a, 1);
  cs.add(b, 2);
  cs.add(zero, 3);
  auto hit = cs.search(c, 1);
  CHECK(hit[0].id == 1 && std::fabs(hit[0].dist) < 1e-6f);

  bool threw = false;
  try {
    Index bad(0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void test_determinism() {
  const std::size_t dim = 16;
  const auto data = clustered(2000, dim, 10, 3);
  const Index a = build(Metric::L2, data, dim), b = build(Metric::L2, data, dim);
  CHECK(a.max_level() == b.max_level());
  for (std::size_t i = 0; i < 50; ++i) {
    const auto x = a.search(data.data() + i * dim, 10, 50), y = b.search(data.data() + i * dim, 10, 50);
    CHECK(x.size() == y.size());
    for (std::size_t j = 0; j < x.size(); ++j) CHECK(x[j].id == y[j].id);
  }
}

void test_save_load(const std::string& dir) {
  const std::size_t dim = 24;
  const auto data = clustered(3000, dim, 20, 4);
  const Index idx = build(Metric::Cosine, data, dim);
  const std::string path = dir + "/vecdex_test.idx";
  idx.save(path);
  const Index back = Index::load(path);
  CHECK(back.size() == idx.size() && back.dim() == dim && back.metric() == Metric::Cosine);
  CHECK(back.m() == idx.m() && back.max_level() == idx.max_level());
  for (std::size_t i = 0; i < 100; ++i) {
    const auto x = idx.search(data.data() + i * dim, 10, 80), y = back.search(data.data() + i * dim, 10, 80);
    CHECK(x.size() == y.size());
    for (std::size_t j = 0; j < x.size(); ++j) CHECK(x[j].id == y[j].id && x[j].dist == y[j].dist);
  }

  auto throws = [](const std::function<void()>& fn) {
    try {
      fn();
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  CHECK(throws([&] { Index::load(dir + "/missing.idx"); }));

  std::ifstream in(path, std::ios::binary);
  std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::string cut = dir + "/vecdex_cut.idx", junk = dir + "/vecdex_junk.idx", flip = dir + "/vecdex_flip.idx";
  std::ofstream(cut, std::ios::binary).write(raw.data(), static_cast<std::streamsize>(raw.size() / 2));
  std::ofstream(junk, std::ios::binary) << "not an index at all";
  std::string bad = raw;
  for (std::size_t i = raw.size() - 64; i < raw.size(); ++i) bad[i] = '\x7f';
  std::ofstream(flip, std::ios::binary).write(bad.data(), static_cast<std::streamsize>(bad.size()));
  CHECK(throws([&] { Index::load(cut); }));
  CHECK(throws([&] { Index::load(junk); }));
  CHECK(throws([&] { Index::load(flip); }));
  for (const auto& p : {path, cut, junk, flip}) std::remove(p.c_str());
}

void test_incremental() {
  const std::size_t dim = 16;
  const auto data = clustered(4000, dim, 16, 5);
  const auto qrys = clustered(100, dim, 16, 5);
  Index idx(dim);
  for (std::size_t i = 0; i < 2000; ++i) idx.add(data.data() + i * dim, i);
  const std::vector<float> half(data.begin(), data.begin() + 2000 * dim);
  CHECK(recall(idx, Metric::L2, half, qrys, dim, 10, 100) >= 0.97);
  for (std::size_t i = 2000; i < 4000; ++i) idx.add(data.data() + i * dim, i);
  CHECK(idx.size() == 4000);
  CHECK(recall(idx, Metric::L2, data, qrys, dim, 10, 100) >= 0.97);
}

}

int main() {
  const char* env = std::getenv("TMPDIR");
  std::string dir = env ? env : "/tmp";
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  struct T {
    const char* name;
    std::function<void()> fn;
  };
  const std::vector<T> all = {
      {"recall", test_recall},
      {"exact_on_small_sets", test_exact_on_small_sets},
      {"edges", test_edges},
      {"determinism", test_determinism},
      {"save_load", [&] { test_save_load(dir); }},
      {"incremental", test_incremental},
  };
  for (const T& t : all) {
    const int before = g_fail;
    std::printf("%s\n", t.name);
    t.fn();
    std::printf("  %s\n", g_fail == before ? "ok" : "FAILED");
  }
  std::printf("%d failure(s)\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
