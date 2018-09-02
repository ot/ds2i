#pragma once

#include "binary_collection.hpp"

namespace ds2i {

class forward_index {
   public:
    forward_index(size_t size) : m_size(size) {}
    size_t size() { return m_size; }
    static forward_index from_binary_collection(const std::string &input_basename);
   private:
    size_t m_size;
};

forward_index forward_index::from_binary_collection(const std::string &input_basename)
{
    binary_collection coll((input_basename + ".docs").c_str());

    auto firstseq = *coll.begin();
    if (firstseq.size() != 1) {
        throw std::invalid_argument("First sequence should only contain number of documents");
    }
    auto num_docs  = *firstseq.begin();
    auto num_terms = std::distance(++coll.begin(), coll.end());

    forward_index fwd(num_docs);

    //uint32_t tid = 0;

    //tqdm::Params params;
    //params.desc="create_forward_index";
    //params.leave = true;
    //params.dynamic_ncols = true;
    //for (auto &&it : tqdm::tqdm(++coll.begin(), coll.end(), params)) {
    //    for (auto &&d : it) {
    //        forward_index[d].doc_id = d;
    //        if (it.size() >= MIN_LEN) {
    //            // TODO: d-gap
    //            Varint::encode_single(tid, forward_index[d].terms);
    //        }
    //    }
    //    ++tid;
    //}

    //std::sort(forward_index.begin(), forward_index.end(), [](auto &&lhs, auto &&rhs) {
    //    if ((lhs.terms.size() == 0 and rhs.terms.size() == 0) or
    //        (lhs.terms.size() != 0 and rhs.terms.size() != 0)) {
    //        return lhs.doc_id < rhs.doc_id;
    //    }
    //    return rhs.terms.size() == 0;
    //});

    return fwd;
}

};
