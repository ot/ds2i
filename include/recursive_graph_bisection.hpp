#pragma once

#include <cmath>
#include <fstream>
#include <iterator>
#include <vector>

#include <pstl/algorithm>
#include <pstl/execution>

#include "binary_collection.hpp"
#include "binary_freq_collection.hpp"
#include "codec/block_codecs.hpp"
#include "codec/varintgb.hpp"
#include "util/index_build_utils.hpp"
#include "util/log.hpp"
#include "util/progress.hpp"
#include "forward_index.hpp"

namespace ds2i {
const Log2<1024> log2;

namespace bp {

inline double expb(double logn1, double logn2, size_t deg1, size_t deg2) {
    double a = deg1 * (logn1 - log2(deg1 + 1));
    double b = deg2 * (logn2 - log2(deg2 + 1));
    return a + b;
};

class precomputed_moves_t {
    using key_type = std::pair<uint32_t, uint32_t>;
    using mapping_type = std::vector<std::vector<uint32_t>>;

   public:
    precomputed_moves_t()                                = default;
    precomputed_moves_t(const precomputed_moves_t &)     = default;
    precomputed_moves_t(precomputed_moves_t &&) noexcept = default;
    precomputed_moves_t &operator=(const precomputed_moves_t &) = default;
    precomputed_moves_t &operator=(precomputed_moves_t &&) noexcept = default;
    precomputed_moves_t(uint32_t size, uint32_t degree_limit) {
        precompute_moves_recursive(size, degree_limit);
    };

    const mapping_type &operator[](const key_type& key) const
    {
        auto pos = m_values.find(key);
        assert(pos != m_values.end());
        return pos->second;
    }

   private:
    void precompute_moves_recursive(uint32_t n, int degree_limit)
    {
        uint32_t n1 = n / 2;
        uint32_t n2 = (n + 1) / 2;
        key_type key(n1, n2);
        auto pos = m_values.find(key);
        if (pos == m_values.end()) {
            auto pre = precompute_moves(log2(n1), log2(n2), degree_limit);
            m_values.insert({key, std::move(pre)});
            precompute_moves_recursive(n1, degree_limit);
            precompute_moves_recursive(n2, degree_limit);
        }
    }

    std::vector<std::vector<uint32_t>> precompute_moves(double   logn1,
                                                        double   logn2,
                                                        uint32_t upper_bound)
    {
        std::vector<std::vector<uint32_t>> precomputed;
        for (uint32_t i = 0; i < upper_bound; i++) {
            precomputed.emplace_back(upper_bound);
            for (uint32_t j = 0; j < upper_bound; j++) {
                precomputed[i][j] = expb(logn1, logn2, i, j);
            }
        }
        return precomputed;
    }

    std::map<key_type, mapping_type> m_values;
};

const precomputed_moves_t precomputed_moves(100000, 1024);

} // namespace bp

struct degree_map_pair {
    std::vector<size_t> left;
    std::vector<size_t> right;
};

template <class Iterator>
struct document_partition;

template <class Iterator>
class document_range {
   public:
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    document_range(Iterator             first,
                   Iterator             last,
                   const forward_index &fwdidx,
                   std::vector<double> &gains)
        : m_first(first), m_last(last), m_fwdidx(fwdidx), m_gains(gains) {}

    Iterator       begin() { return m_first; }
    Iterator       end() { return m_last; }
    std::ptrdiff_t size() const { return std::distance(m_first, m_last); }

    document_partition<Iterator> split() const {
        Iterator mid = std::next(m_first, size() / 2);
        return {document_range(m_first, mid, m_fwdidx, m_gains),
                document_range(mid, m_last, m_fwdidx, m_gains),
                term_count()};
    }

    std::size_t           term_count() const { return m_fwdidx.term_count(); }
    std::vector<uint32_t> terms(value_type document) const { return m_fwdidx.terms(document); }
    double                gain(value_type document) const { return m_gains[document]; }
    double &              gain(value_type document) { return m_gains[document]; }

    auto by_gain() {
        return [this](const value_type &lhs, const value_type &rhs) {
            return m_gains[lhs] > m_gains[rhs];
        };
    }

   private:
    Iterator             m_first;
    Iterator             m_last;
    const forward_index &m_fwdidx;
    std::vector<double> &m_gains;
};

template <class Iterator>
struct document_partition {
    document_range<Iterator> left;
    document_range<Iterator> right;
    size_t                   term_count;
};

auto get_mapping = [](const auto &collection) {
    std::vector<uint32_t> mapping(collection.size(), 0u);
    size_t                p = 0;
    for (const auto &id : collection) {
        mapping[id] = p++;
    }
    return mapping;
};

template <class Iterator>
std::vector<size_t> compute_degrees(document_range<Iterator> &range) {
    std::vector<size_t> deg_map(range.term_count());
    for (const auto &document : range) {
        for (const auto &term : range.terms(document)) {
            deg_map[term] += 1;
        }
    }
    return deg_map;
}

template <class Iterator>
degree_map_pair compute_degrees(document_partition<Iterator> &partition) {
    std::vector<size_t> left_degree;
    std::vector<size_t> right_degree;
    tbb::parallel_invoke([&] { left_degree = compute_degrees(partition.left); },
                         [&] { right_degree = compute_degrees(partition.right); });
    return degree_map_pair{left_degree, right_degree};
}

template <typename Iter>
using gain_function_t = std::function<void(document_range<Iter> &,
                                           const std::ptrdiff_t,
                                           const std::ptrdiff_t,
                                           const std::vector<size_t> &,
                                           const std::vector<size_t> &)>;

template<class T>
class cache_entry {
   public:
    cache_entry() : m_value(), m_has_value(false) {}

    const T &value() { return m_value; }
    bool     has_value() { return m_has_value; }
    void     operator=(const T &v) {
        m_value     = v;
        m_has_value = true;
    }

   private:
    T m_value;
    bool m_has_value;
};

template <typename Iter>
void compute_move_gains_precompute(document_range<Iter> &     range,
                                   const std::ptrdiff_t       from_n,
                                   const std::ptrdiff_t       to_n,
                                   const std::vector<size_t> &from_lex,
                                   const std::vector<size_t> &to_lex) {
    const auto logn1 = log2(from_n);
    const auto logn2 = log2(to_n);

    const auto& precomputed = bp::precomputed_moves[std::make_pair(from_n, to_n)];
    assert(not precomputed.empty());
    auto compute_document_gain = [&](auto &d) {
        double gain = 0.0;
        for (const auto &t : range.terms(d)) {
            auto from_deg = from_lex[t];
            auto to_deg   = to_lex[t];
            gain += precomputed[from_deg][to_deg];
        }
        range.gain(d) = gain;
    };
    std::for_each(std::execution::par_unseq, range.begin(), range.end(), compute_document_gain);
}

template <typename Iter>
void compute_move_gains_caching(document_range<Iter> &     range,
                                const std::ptrdiff_t       from_n,
                                const std::ptrdiff_t       to_n,
                                const std::vector<size_t> &from_lex,
                                const std::vector<size_t> &to_lex) {
    const auto logn1 = log2(from_n);
    const auto logn2 = log2(to_n);

    std::vector<cache_entry<double>> gain_cache(from_lex.size());
    auto       compute_document_gain = [&](auto &d) {
        double gain = 0.0;
        for (const auto &t : range.terms(d)) {
            if (not gain_cache[t].has_value()) {
                auto from_deg = from_lex[t];
                auto to_deg   = to_lex[t];
                auto term_gain = bp::expb(logn1, logn2, from_deg, to_deg) -
                                 bp::expb(logn1, logn2, from_deg - 1, to_deg + 1);
                gain_cache[t] = term_gain;
            }
            gain += gain_cache[t].value();
        }
        range.gain(d) = gain;
    };
    std::for_each(range.begin(), range.end(), compute_document_gain);
}

template <typename Iter>
void compute_move_gains(document_range<Iter> &     range,
                        const std::ptrdiff_t       from_n,
                        const std::ptrdiff_t       to_n,
                        const std::vector<size_t> &from_lex,
                        const std::vector<size_t> &to_lex) {
    const auto logn1 = log2(from_n);
    const auto logn2 = log2(to_n);

    auto compute_document_gain = [&](const auto &d) {
        double gain = 0.0;
        auto   terms = range.terms(d);
        for (const auto &t : terms) {
            auto from_deg = from_lex[t];
            auto to_deg   = to_lex[t];
            auto term_gain = bp::expb(logn1, logn2, from_deg, to_deg) -
                             bp::expb(logn1, logn2, from_deg - 1, to_deg + 1);
            gain += term_gain;
        }
        range.gain(d) = gain;
    };
    std::for_each(range.begin(), range.end(), compute_document_gain);
}

template <class Iterator>
void compute_gains(document_partition<Iterator> &partition,
                   const degree_map_pair &       degrees,
                   gain_function_t<Iterator>     gain_function) {

    auto n1 = partition.left.size();
    auto n2 = partition.right.size();
    tbb::parallel_invoke(
        [&] { gain_function(partition.left, n1, n2, degrees.left, degrees.right); },
        [&] { gain_function(partition.right, n2, n1, degrees.right, degrees.left); });
}

template <class Iterator>
void swap(document_partition<Iterator> &partition, degree_map_pair &degrees) {
    auto left  = partition.left;
    auto right = partition.right;
    auto lit   = left.begin();
    auto rit   = right.begin();
    for (; lit != left.end() && rit != right.end(); ++lit, ++rit) {
        if (left.gain(*lit) + right.gain(*rit) <= 0) {
            break;
        }
        for (auto &term : left.terms(*lit)) {
            degrees.left[term]--;
            degrees.right[term]++;
        }
        for (auto &term : right.terms(*rit)) {
            degrees.left[term]++;
            degrees.right[term]--;
        }
        std::iter_swap(lit, rit);
    }
}

template <class Iterator>
void process_partition(document_partition<Iterator> &partition,
                       gain_function_t<Iterator>     gain_function) {

    auto degrees = compute_degrees(partition);
    for (int iteration = 0; iteration < 20; ++iteration) {
        compute_gains(partition, degrees, gain_function);
        tbb::parallel_invoke(
            [&] {
                std::sort(std::execution::par_unseq,
                          partition.left.begin(),
                          partition.left.end(),
                          partition.left.by_gain());
            },
            [&] {
                std::sort(std::execution::par_unseq,
                          partition.right.begin(),
                          partition.right.end(),
                          partition.right.by_gain());
            });
        swap(partition, degrees);
    }
}

template <class Iterator>
void recursive_graph_bisection(document_range<Iterator> documents, int depth, progress &p) {
    auto partition = documents.split();
    if (documents.size() >= 1024) {
        process_partition(partition, compute_move_gains_caching<Iterator>);
    } else {
        process_partition(partition, compute_move_gains<Iterator>);
    }
    p.update(documents.size());
    if (depth > 1 && documents.size() > 2) {
        tbb::parallel_invoke([&] { recursive_graph_bisection(partition.left, depth - 1, p); },
                             [&] { recursive_graph_bisection(partition.right, depth - 1, p); });
    } else {
        std::sort(partition.left.begin(), partition.left.end());
        std::sort(partition.right.begin(), partition.right.end());
    }
}

} // namespace ds2i
