#pragma once

#include <cmath>
#include <vector>
#include <fstream>

#include <pstl/algorithm>
#include <pstl/execution>

#include "binary_collection.hpp"
#include "binary_freq_collection.hpp"
#include "codec/block_codecs.hpp"
#include "util/index_build_utils.hpp"
#include "util/log.hpp"
#include "util/progress.hpp"

namespace ds2i {

constexpr int MIN_LEN = 1;
constexpr int PARALLEL_THRESHOLD = 32;
const Log2<1024> log2;

namespace bp {

inline double expb(double logn1, double logn2, size_t deg1, size_t deg2)
{
    double a = deg1 * (logn1 - log2(deg1 + 1));
    double b = deg2 * (logn2 - log2(deg2 + 1));
    return a + b;
};

struct doc_entry {
    uint32_t              id    = 0;
    int                   range = 0;
    double                gain  = 0.0;
    std::vector<uint8_t>  terms_compressed{};
    std::vector<uint32_t> terms() const
    {
        std::vector<uint32_t> terms;
        terms.resize(terms_compressed.size() * 5);
        size_t n = 0;
        TightVariableByte::decode(terms_compressed.data(), terms.data(), terms_compressed.size(), n);
        for (auto it = std::next(terms.begin()); it != terms.end(); ++it) {
            *it += *std::prev(it);
        }
        terms.resize(n);
        terms.shrink_to_fit();
        return terms;
    }
};

class forward_index {
   public:
    forward_index(size_t size, size_t term_count)
        : m_size(size), m_term_count(term_count), m_documents(size) {}
    size_t               size() { return m_size; }
    size_t               term_count() { return m_term_count; }
    static forward_index from_binary_collection(const std::string &input_basename);
    doc_entry &          operator[](size_t n) { return m_documents[n]; }
    const doc_entry &    operator[](size_t n) const { return m_documents[n]; }
    auto                 begin() { return m_documents.begin(); }
    auto                 end() { return m_documents.end(); }

   private:
    size_t                 m_size;
    size_t                 m_term_count;
    std::vector<doc_entry> m_documents;
};

forward_index forward_index::from_binary_collection(const std::string &input_basename) {
    binary_collection coll((input_basename + ".docs").c_str());

    auto firstseq = *coll.begin();
    if (firstseq.size() != 1) {
        throw std::invalid_argument("First sequence should only contain number of documents");
    }
    auto num_docs  = *firstseq.begin();
    auto num_terms = std::distance(++coll.begin(), coll.end());

    forward_index fwd(num_docs, num_terms);
    progress p("Building forward index", num_terms);

    uint32_t tid = 0;

    std::vector<uint32_t> prev(num_docs, 0u);
    for (auto it = ++coll.begin(); it != coll.end(); ++it) {
        for (const auto &d : *it) {
            fwd[d].id = d;
            if (it->size() >= MIN_LEN) {
                TightVariableByte::encode_single(tid - prev[d], fwd[d].terms_compressed);
                prev[d] = tid;
            }
        }
        p.update_and_print(1);
        ++tid;
    }

    return fwd;
}
} // namespace bp

struct doc_ref {
    explicit doc_ref(bp::doc_entry *d) : ref(d) {}
    doc_ref()                = default;
    doc_ref(const doc_ref &) = default;
    doc_ref(doc_ref &&)      = default;
    doc_ref &operator=(const doc_ref &) = default;
    doc_ref &operator=(doc_ref &&) = default;
    ~doc_ref()                     = default;

    bp::doc_entry *ref;

    uint32_t id() const { return ref->id; }
    int      range() const { return ref->range; }
    double   gain() const { return ref->gain; }
    void     update_gain(double gain) { ref->gain = gain; }
    void     update_range(int range) { ref->range = range; }

    std::vector<uint32_t> terms() const { return ref->terms(); }

    static auto by_range_gain() {
        return [](const doc_ref &lhs, const doc_ref &rhs) {
            // TODO: make more efficient
            return std::make_pair(lhs.range(), -lhs.gain()) <
                   std::make_pair(rhs.range(), -rhs.gain());
        };
    }

    static auto by_gain() {
        return [](const doc_ref &lhs, const doc_ref &rhs) { return lhs.gain() > rhs.gain(); };
    }
    static auto by_id() {
        return [](const doc_ref &lhs, const doc_ref &rhs) { return lhs.id() < rhs.id(); };
    }
};

template <class Iterator>
constexpr void assert_random_access() {
    static_assert(std::iterator_traits<Iterator>::iterator_category,
                  "iterator must be random access");
}

struct degree_map_pair {
    std::vector<size_t> left;
    std::vector<size_t> right;
};

template <class Iterator>
struct document_partition;

template <class Iterator>
struct document_range {
    int      id;
    Iterator first;
    Iterator last;
    size_t   term_count;

    Iterator       begin() { return first; }
    Iterator       end() { return last; }
    std::ptrdiff_t size() const { return std::distance(first, last); }

    document_partition<Iterator> split() const {
        auto     left_id  = 2 * id;
        auto     right_id = left_id + 1;
        Iterator mid      = std::next(first, std::distance(first, last) / 2);
        return {
            {left_id, first, mid, term_count}, {right_id, mid, last, term_count}, {}, term_count};
    }

    void update_range_ids() {
        for (auto &doc : *this) {
            doc.update_range(id);
        }
    }
};

template<class Iterator>
struct document_partition {
    document_range<Iterator> left;
    document_range<Iterator> right;
    degree_map_pair          degrees{};
    size_t                   term_count;
};

auto get_mapping = [](const auto &collection)
{
    std::vector<uint32_t> mapping(collection.size(), 0u);
    size_t p = 0;
    for (const auto &i : collection) {
        mapping[i.id()] = p++;
    }
    return mapping;
};

template <class Iterator>
std::vector<size_t> compute_degrees(document_range<Iterator> &range) {
    std::vector<size_t> deg_map(range.term_count);
    for (const auto &document : range) {
        for (const auto &term : document.terms()) {
            deg_map[term] += 1;
        }
    }
    return deg_map;
}

template <class Iterator>
void compute_degrees(document_partition<Iterator> &partition) {
    partition.degrees =
        degree_map_pair{compute_degrees(partition.left), compute_degrees(partition.right)};
}

template <typename Iter>
void compute_move_gains(Iter                       begin,
                        Iter                       end,
                        const std::ptrdiff_t       from_n,
                        const std::ptrdiff_t       to_n,
                        const std::vector<size_t> &from_lex,
                        const std::vector<size_t> &to_lex) {
    const auto logn1 = log2(from_n);
    const auto logn2 = log2(to_n);
    auto compute_document_gain = [&](auto& d) {
        double gain  = 0.0;
        for(const auto& t : d.terms()) {
            auto from_deg = from_lex[t];
            auto to_deg = to_lex[t];
            if (from_deg > 0) {
                gain += bp::expb(logn1, logn2, from_deg, to_deg);
                gain -= bp::expb(logn1, logn2, from_deg - 1, to_deg + 1);
            }
        }
        return d.update_gain(gain);
    };
    std::for_each(std::execution::par_unseq, begin, end, compute_document_gain);
}

template <class Iterator>
void compute_gains(document_partition<Iterator> &partition) {
    auto n1 = partition.left.size();
    auto n2 = partition.right.size();
    compute_move_gains(partition.left.begin(),
                       partition.left.end(),
                       n1,
                       n2,
                       partition.degrees.left,
                       partition.degrees.right);
    compute_move_gains(partition.right.begin(),
                       partition.right.end(),
                       n2,
                       n1,
                       partition.degrees.right,
                       partition.degrees.left);
}

template <class Iterator>
void swap(document_partition<Iterator> &partition) {
    auto left  = partition.left.begin();
    auto right = partition.right.begin();
    for (; left != partition.left.end() && right != partition.right.end(); ++left, ++right) {
        if (left->gain() + right->gain() <= 0) {
            break;
        }
        for (auto &term : left->terms()) {
            partition.degrees.left[term]--;
            partition.degrees.right[term]++;
        }
        for (auto &term : right->terms()) {
            partition.degrees.left[term]++;
            partition.degrees.right[term]--;
        }
        std::iter_swap(left, right);
    }
}

template <class Iterator>
void process_partition(document_partition<Iterator>& partition)
{
    compute_degrees(partition);
    for (int iteration = 0; iteration < 20; ++iteration) {
        compute_gains(partition);
        std::sort(std::execution::par_unseq,
                  partition.left.begin(),
                  partition.left.end(),
                  doc_ref::by_gain());
        std::sort(std::execution::par_unseq,
                  partition.right.begin(),
                  partition.right.end(),
                  doc_ref::by_gain());
        swap(partition);
    }
}

template <class Iterator>
void recursive_graph_bisection(document_range<Iterator> documents, int depth, progress& p)
{
    auto partition = documents.split();
    process_partition(partition);
    p.update_and_print(documents.size());
    if (depth > 1) {
        recursive_graph_bisection(partition.left, depth - 1, p);
        recursive_graph_bisection(partition.right, depth - 1, p);
    }
}

template <class Iterator>
struct recursive_graph_bisection_f {
    recursive_graph_bisection_f(document_range<Iterator> documents, int depth, progress &p)
        : documents(documents), depth(depth), p(p) {}
    void operator()() const {
        auto partition = documents.split();
        process_partition(partition);
        p.update_and_print(documents.size());
        if (depth > 1) {
            recursive_graph_bisection_f<Iterator> left(partition.left, depth - 1, p);
            recursive_graph_bisection_f<Iterator> right(partition.right, depth - 1, p);
            tbb::parallel_invoke(left, right);
        }
    }

    document_range<Iterator> documents;
    int                      depth;
    progress &               p;

};

template <class Iterator>
void recursive_graph_bisection_mt(document_range<Iterator> documents, int depth, progress &p)
{
    recursive_graph_bisection_f<Iterator> root(documents, depth, p);
    root();
}

} // namespace ds2i
