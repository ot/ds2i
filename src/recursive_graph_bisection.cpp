#include "CLI/CLI.hpp"

#include <pstl/algorithm>
#include <pstl/execution>

#include "recursive_graph_bisection.hpp"

int main(int argc, char const *argv[]) {

    std::string input_basename;
    std::string output_basename;
    size_t      min_len;

    CLI::App app{"Recursive graph bisection algorithm used for inverted indexed reordering."};
    app.add_option("-c,--collection", input_basename, "Collection basename")->required();
    app.add_option("-o,--output", output_basename, "Output basename")->required();
    app.add_option("-m,--min-len", min_len, "Minimum list threshold");
    CLI11_PARSE(app, argc, argv);

    using namespace ds2i;

    bp::forward_index fwd = bp::forward_index::from_binary_collection(input_basename);

    std::vector<doc_ref> documents;
    std::transform(
        fwd.begin(), fwd.end(), std::back_inserter(documents), [](auto &d) { return doc_ref(&d); });
    document_range<std::vector<doc_ref>::iterator> initial_range{
        0, documents.begin(), documents.end(), fwd.term_count()};
    recursive_graph_bisection(initial_range, 9);
    auto mapping = get_mapping(documents);
    //std::vector<uint32_t> mapping(fwd.size(), 0u);
    //std::iota(mapping.begin(), mapping.end(), 0u);
    reorder_inverted_index(input_basename, output_basename, mapping);

    return 0;
}
