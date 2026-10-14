#include "vecdex/index.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace vecdex {

namespace {

constexpr std::uint32_t kMagic = 0x58444356;
constexpr std::uint32_t kVersion = 1;

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && (defined(__aarch64__) || defined(_M_ARM64))

float l2(const float* a, const float* b, std::size_t n) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  for (; i + 4 <= n; i += 4) {
    const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, d, d);
  }
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float dot(const float* a, const float* b, std::size_t n) {
  float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

#elif defined(__AVX2__) && defined(__FMA__)

float hsum(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

float l2(const float* a, const float* b, std::size_t n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
  }
  for (; i + 8 <= n; i += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(d, d, s0);
  }
  float acc = hsum(_mm256_add_ps(s0, s1));
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float dot(const float* a, const float* b, std::size_t n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  float acc = hsum(_mm256_add_ps(s0, s1));
  for (; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

#else

float l2(const float* a, const float* b, std::size_t n) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1];
    const float d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
    s0 += d0 * d0;
    s1 += d1 * d1;
    s2 += d2 * d2;
    s3 += d3 * d3;
  }
  float acc = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) {
    const float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

float dot(const float* a, const float* b, std::size_t n) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  std::size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  float acc = (s0 + s1) + (s2 + s3);
  for (; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

#endif

struct Visited {
  std::vector<std::uint32_t> tag;
  std::uint32_t epoch = 0;

  void reset(std::size_t n) {
    if (tag.size() < n) tag.resize(n, 0);
    if (++epoch == 0) {
      std::fill(tag.begin(), tag.end(), 0);
      epoch = 1;
    }
  }
  bool test_set(std::uint32_t i) {
    if (tag[i] == epoch) return true;
    tag[i] = epoch;
    return false;
  }
};

thread_local Visited tl_seen;

template <typename T>
void put(std::ofstream& out, const T& v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
void put(std::ofstream& out, const std::vector<T>& v) {
  put(out, static_cast<std::uint64_t>(v.size()));
  out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template <typename T>
void get(std::ifstream& in, T& v) {
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("vecdex: truncated index file");
}

template <typename T>
void get(std::ifstream& in, std::vector<T>& v, std::uint64_t limit) {
  std::uint64_t n = 0;
  get(in, n);
  if (n > limit) throw std::runtime_error("vecdex: corrupt index file");
  v.resize(n);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (!in) throw std::runtime_error("vecdex: truncated index file");
}

}

Index::Index(std::size_t dim, Metric mt, std::size_t m, std::size_t ef_build, std::uint64_t seed)
    : dim_(dim),
      mt_(mt),
      m_(m),
      m0_(m * 2),
      ef_build_(std::max(ef_build, m)),
      mult_(1.0 / std::log(static_cast<double>(m))),
      rng_(seed) {
  if (dim == 0) throw std::invalid_argument("vecdex: dim must be positive");
  if (m < 2) throw std::invalid_argument("vecdex: m must be at least 2");
}

void Index::reserve(std::size_t n) {
  data_.reserve(n * dim_);
  ids_.reserve(n);
  lvl_.reserve(n);
  base_.reserve(n * (m0_ + 1));
  upper_.reserve(n);
}

float Index::dist(const float* a, const float* b) const {
  return mt_ == Metric::L2 ? l2(a, b, dim_) : 1.0f - dot(a, b, dim_);
}

void Index::normalize(float* v) const {
  const float nrm = std::sqrt(dot(v, v, dim_));
  if (nrm > 0.0f) {
    const float inv = 1.0f / nrm;
    for (std::size_t i = 0; i < dim_; ++i) v[i] *= inv;
  }
}

Index::Node* Index::links(Node n, int lvl) {
  if (lvl == 0) return base_.data() + static_cast<std::size_t>(n) * (m0_ + 1);
  return upper_[n].data() + static_cast<std::size_t>(lvl - 1) * (m_ + 1);
}

const Index::Node* Index::links(Node n, int lvl) const {
  if (lvl == 0) return base_.data() + static_cast<std::size_t>(n) * (m0_ + 1);
  return upper_[n].data() + static_cast<std::size_t>(lvl - 1) * (m_ + 1);
}

Index::Node Index::greedy(const float* qry, Node cur, int lvl) const {
  float best = dist(qry, vec(cur));
  bool moved = true;
  while (moved) {
    moved = false;
    const Node* lk = links(cur, lvl);
    const Node cnt = lk[0];
    for (Node i = 1; i <= cnt; ++i) {
      const float d = dist(qry, vec(lk[i]));
      if (d < best) {
        best = d;
        cur = lk[i];
        moved = true;
      }
    }
  }
  return cur;
}

std::vector<Index::Cand> Index::search_layer(const float* qry, Node ep, std::size_t ef,
                                             int lvl) const {
  Visited& seen = tl_seen;
  seen.reset(ids_.size());

  std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> open;
  std::priority_queue<Cand> best;

  const float d0 = dist(qry, vec(ep));
  open.emplace(d0, ep);
  best.emplace(d0, ep);
  seen.test_set(ep);

  while (!open.empty()) {
    const Cand cur = open.top();
    if (cur.first > best.top().first && best.size() >= ef) break;
    open.pop();
    const Node* lk = links(cur.second, lvl);
    const Node cnt = lk[0];
    if (cnt > 0) __builtin_prefetch(vec(lk[1]));
    for (Node i = 1; i <= cnt; ++i) {
      const Node nb = lk[i];
      if (i < cnt) __builtin_prefetch(vec(lk[i + 1]));
      if (seen.test_set(nb)) continue;
      const float d = dist(qry, vec(nb));
      if (best.size() < ef || d < best.top().first) {
        open.emplace(d, nb);
        best.emplace(d, nb);
        if (best.size() > ef) best.pop();
      }
    }
  }

  std::vector<Cand> out(best.size());
  for (std::size_t i = out.size(); i-- > 0;) {
    out[i] = best.top();
    best.pop();
  }
  return out;
}

std::vector<Index::Node> Index::select(const std::vector<Cand>& cands, std::size_t want) const {
  std::vector<Node> picks;
  picks.reserve(want);
  for (const Cand& c : cands) {
    if (picks.size() >= want) break;
    bool keep = true;
    for (const Node p : picks) {
      if (dist(vec(c.second), vec(p)) < c.first) {
        keep = false;
        break;
      }
    }
    if (keep) picks.push_back(c.second);
  }
  return picks;
}

void Index::connect(Node src, const std::vector<Node>& picks, int lvl) {
  Node* own = links(src, lvl);
  own[0] = static_cast<Node>(picks.size());
  std::copy(picks.begin(), picks.end(), own + 1);

  const std::size_t lim = cap(lvl);
  for (const Node nb : picks) {
    Node* lk = links(nb, lvl);
    if (lk[0] < lim) {
      lk[++lk[0]] = src;
      continue;
    }
    std::vector<Cand> pool;
    pool.reserve(lim + 1);
    pool.emplace_back(dist(vec(nb), vec(src)), src);
    for (Node i = 1; i <= lk[0]; ++i) pool.emplace_back(dist(vec(nb), vec(lk[i])), lk[i]);
    std::sort(pool.begin(), pool.end());
    const std::vector<Node> kept = select(pool, lim);
    lk[0] = static_cast<Node>(kept.size());
    std::copy(kept.begin(), kept.end(), lk + 1);
  }
}

void Index::add(const float* v, std::uint64_t id) {
  if (ids_.size() >= std::numeric_limits<Node>::max()) throw std::length_error("vecdex: index is full");
  const Node cur = static_cast<Node>(ids_.size());

  data_.insert(data_.end(), v, v + dim_);
  float* own = data_.data() + static_cast<std::size_t>(cur) * dim_;
  if (mt_ == Metric::Cosine) normalize(own);

  std::uniform_real_distribution<double> uni(std::numeric_limits<double>::min(), 1.0);
  const int lvl = static_cast<int>(-std::log(uni(rng_)) * mult_);

  ids_.push_back(id);
  lvl_.push_back(lvl);
  base_.resize(base_.size() + m0_ + 1, 0);
  upper_.emplace_back(static_cast<std::size_t>(lvl) * (m_ + 1), 0);

  if (cur == 0) {
    entry_ = 0;
    max_lvl_ = lvl;
    return;
  }

  Node ep = entry_;
  for (int l = max_lvl_; l > lvl; --l) ep = greedy(own, ep, l);

  for (int l = std::min(lvl, max_lvl_); l >= 0; --l) {
    const std::vector<Cand> found = search_layer(own, ep, ef_build_, l);
    connect(cur, select(found, m_), l);
    ep = found.front().second;
    own = data_.data() + static_cast<std::size_t>(cur) * dim_;
  }

  if (lvl > max_lvl_) {
    max_lvl_ = lvl;
    entry_ = cur;
  }
}

std::vector<Hit> Index::search(const float* qry, std::size_t k, std::size_t ef) const {
  std::vector<Hit> out;
  if (ids_.empty() || k == 0) return out;

  std::vector<float> tmp;
  if (mt_ == Metric::Cosine) {
    tmp.assign(qry, qry + dim_);
    normalize(tmp.data());
    qry = tmp.data();
  }

  Node ep = entry_;
  for (int l = max_lvl_; l > 0; --l) ep = greedy(qry, ep, l);

  const std::vector<Cand> found = search_layer(qry, ep, std::max(ef, k), 0);
  const std::size_t n = std::min(k, found.size());
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) out.push_back({found[i].first, ids_[found[i].second]});
  return out;
}

void Index::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("vecdex: cannot open " + path);
  put(out, kMagic);
  put(out, kVersion);
  put(out, static_cast<std::uint64_t>(dim_));
  put(out, static_cast<std::uint32_t>(mt_));
  put(out, static_cast<std::uint64_t>(m_));
  put(out, static_cast<std::uint64_t>(ef_build_));
  put(out, static_cast<std::int32_t>(max_lvl_));
  put(out, entry_);
  put(out, data_);
  put(out, ids_);
  put(out, lvl_);
  put(out, base_);
  for (const auto& up : upper_) put(out, up);
  if (!out) throw std::runtime_error("vecdex: write failed for " + path);
}

Index Index::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("vecdex: cannot open " + path);
  in.seekg(0, std::ios::end);
  const std::uint64_t bytes = static_cast<std::uint64_t>(in.tellg());
  in.seekg(0);

  std::uint32_t magic = 0, ver = 0, mt = 0;
  std::uint64_t dim = 0, m = 0, ef = 0;
  std::int32_t max_lvl = 0;
  get(in, magic);
  get(in, ver);
  if (magic != kMagic) throw std::runtime_error("vecdex: not an index file");
  if (ver != kVersion) throw std::runtime_error("vecdex: unsupported index version");
  get(in, dim);
  get(in, mt);
  get(in, m);
  get(in, ef);
  get(in, max_lvl);
  if (dim == 0 || dim > bytes || m < 2 || m > bytes || mt > 2) {
    throw std::runtime_error("vecdex: corrupt index file");
  }

  Index idx(dim, static_cast<Metric>(mt), m, ef);
  idx.max_lvl_ = max_lvl;
  get(in, idx.entry_);
  get(in, idx.data_, bytes);
  get(in, idx.ids_, bytes);
  get(in, idx.lvl_, bytes);
  get(in, idx.base_, bytes);

  const std::size_t n = idx.ids_.size();
  if (idx.data_.size() != n * idx.dim_ || idx.lvl_.size() != n ||
      idx.base_.size() != n * (idx.m0_ + 1) || (n > 0 && idx.entry_ >= n)) {
    throw std::runtime_error("vecdex: corrupt index file");
  }
  idx.upper_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    get(in, idx.upper_[i], bytes);
    if (idx.lvl_[i] < 0 || idx.upper_[i].size() != static_cast<std::size_t>(idx.lvl_[i]) * (idx.m_ + 1)) {
      throw std::runtime_error("vecdex: corrupt index file");
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (int l = 0; l <= idx.lvl_[i]; ++l) {
      const Node* lk = idx.links(static_cast<Node>(i), l);
      if (lk[0] > idx.cap(l)) throw std::runtime_error("vecdex: corrupt index file");
      for (Node j = 1; j <= lk[0]; ++j) {
        if (lk[j] >= n || idx.lvl_[lk[j]] < l) throw std::runtime_error("vecdex: corrupt index file");
      }
    }
  }
  return idx;
}

}
