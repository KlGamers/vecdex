#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace vecdex {

enum class Metric : std::uint32_t { L2 = 0, IP = 1, Cosine = 2 };

struct Hit {
  float dist;
  std::uint64_t id;
};

class Index {
 public:
  explicit Index(std::size_t dim, Metric mt = Metric::L2, std::size_t m = 16,
                 std::size_t ef_build = 200, std::uint64_t seed = 42);

  void reserve(std::size_t n);
  void add(const float* vec, std::uint64_t id);
  std::vector<Hit> search(const float* qry, std::size_t k, std::size_t ef = 0) const;

  void save(const std::string& path) const;
  static Index load(const std::string& path);

  std::size_t size() const { return ids_.size(); }
  std::size_t dim() const { return dim_; }
  std::size_t m() const { return m_; }
  std::size_t ef_build() const { return ef_build_; }
  Metric metric() const { return mt_; }
  int max_level() const { return max_lvl_; }

 private:
  using Node = std::uint32_t;
  using Cand = std::pair<float, Node>;

  float dist(const float* a, const float* b) const;
  const float* vec(Node n) const { return data_.data() + static_cast<std::size_t>(n) * dim_; }
  Node* links(Node n, int lvl);
  const Node* links(Node n, int lvl) const;
  std::size_t cap(int lvl) const { return lvl == 0 ? m0_ : m_; }

  Node greedy(const float* qry, Node cur, int lvl) const;
  std::vector<Cand> search_layer(const float* qry, Node ep, std::size_t ef, int lvl) const;
  std::vector<Node> select(const std::vector<Cand>& cands, std::size_t want) const;
  void connect(Node src, const std::vector<Node>& picks, int lvl);
  void normalize(float* v) const;

  std::size_t dim_;
  Metric mt_;
  std::size_t m_;
  std::size_t m0_;
  std::size_t ef_build_;
  double mult_;
  std::mt19937_64 rng_;

  std::vector<float> data_;
  std::vector<std::uint64_t> ids_;
  std::vector<int> lvl_;
  std::vector<Node> base_;
  std::vector<std::vector<Node>> upper_;
  int max_lvl_ = -1;
  Node entry_ = 0;
};

}
