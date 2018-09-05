#include "CLI/CLI.hpp"
#include "pstl/execution"
#include "tbb/task_scheduler_init.h"

#include "recursive_graph_bisection.hpp"
#include "util/progress.hpp"

int main(int argc, char const *argv[]) {

    std::string input_basename;
    std::string output_basename;
    size_t      min_len = 0;
    size_t      depth   = 0;
    size_t      threads = 4;

    CLI::App app{"Recursive graph bisection algorithm used for inverted indexed reordering."};
    app.add_option("-c,--collection", input_basename, "Collection basename")->required();
    app.add_option("-o,--output", output_basename, "Output basename")->required();
    app.add_option("-m,--min-len", min_len, "Minimum list threshold");
    app.add_option("-d,--depth", depth, "Recursion depth");
    app.add_option("-t,--threads", threads, "Thread count");
    CLI11_PARSE(app, argc, argv);

    tbb::task_scheduler_init init(threads);

    using namespace ds2i;

    bp::forward_index fwd = bp::forward_index::from_binary_collection(input_basename, min_len);
    std::vector<doc_ref> documents;
    std::transform(
        fwd.begin(), fwd.end(), std::back_inserter(documents), [](auto &d) { return doc_ref(&d); });
    document_range<std::vector<doc_ref>::iterator> initial_range{
        0, documents.begin(), documents.end(), fwd.term_count()};

    if (depth == 0u) {
        depth = static_cast<size_t>(std::log2(fwd.size()));
    }
    std::cerr << "Using max depth " << depth << std::endl;
    ds2i::progress bp_progress("Graph bisection", initial_range.size() * depth);
    bp_progress.update_and_print(0);
    recursive_graph_bisection(initial_range, depth, bp_progress);

    auto mapping = get_mapping(documents);
    fwd.clear();
    documents.clear();
    reorder_inverted_index(input_basename, output_basename, mapping);
    return 0;
}
