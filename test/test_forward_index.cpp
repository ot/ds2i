#define BOOST_TEST_MODULE forward_index

#include "test_generic_sequence.hpp"

#include "codec/block_codecs.hpp"
#include "codec/maskedvbyte.hpp"
#include "codec/streamvbyte.hpp"
#include "codec/qmx.hpp"
#include "codec/varintgb.hpp"
#include "codec/simple8b.hpp"
#include "codec/simple16.hpp"
#include "codec/simdbp.hpp"

#include "block_posting_list.hpp"

#include <vector>
#include <cstdlib>
#include <algorithm>
#include "test_generic_sequence.hpp"

#include "recursive_graph_bisection.hpp"

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <numeric>

BOOST_AUTO_TEST_CASE(write_and_read)
{
    // given
    using namespace ds2i;
    std::string invind_input("test_data/test_collection");
    std::string fwdind_file("temp_collection");
    auto fwd = bp::forward_index::from_inverted_index(invind_input, 0);

    // when
    bp::forward_index::write(fwd, fwdind_file);
    auto fwd_read = bp::forward_index::read(fwdind_file);

    //// then
    BOOST_REQUIRE_EQUAL(fwd.size(), fwd_read.size());
    BOOST_REQUIRE_EQUAL(fwd.term_count(), fwd_read.term_count());
    for (size_t doc = 0; doc < fwd.size(); ++doc) {
        BOOST_REQUIRE_EQUAL(fwd[doc].id, fwd_read[doc].id);
        BOOST_REQUIRE_EQUAL(fwd[doc].term_count, fwd_read[doc].term_count);
        BOOST_CHECK_EQUAL_COLLECTIONS(fwd[doc].terms_compressed.begin(),
                                      fwd[doc].terms_compressed.end(),
                                      fwd_read[doc].terms_compressed.begin(),
                                      fwd_read[doc].terms_compressed.end());
    }
}
