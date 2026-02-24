# include "Graph.hpp"

void convert2ligra(const Graph &g, const std::string &output_file) {
    const size_t n = g.get_vertex_num();
    const size_t m = g.get_edge_num();
    auto csr_vlist = g.get_csr_vlist();
    auto csr_elist = g.get_csr_elist();
    
    std::ofstream output(output_file);
    output << "AdjacencyGraph" << std::endl;
    output << n << std::endl;
    output << m << std::endl;
    for (int i = 0; i < n; ++i) {
        output << csr_vlist[i] << std::endl;
    }
    for (int j = 0; j < m; ++j) {
        output << csr_elist[j] << std::endl;
    }
    output.close();
}

void convert2wligra(const Graph &g, const std::string &output_file) {
    const size_t n = g.get_vertex_num();
    const size_t m = g.get_edge_num();
    auto csr_vlist = g.get_csr_vlist();
    auto csr_elist = g.get_csr_elist();
    auto csr_weight = g.get_csr_weight();
    
    std::ofstream output(output_file);
    output << "WeightedAdjacencyGraph" << std::endl;
    output << n << std::endl;
    output << m << std::endl;
    for (int i = 0; i < n; ++i) {
        output << csr_vlist[i] << std::endl;
    }
    for (int j = 0; j < m; ++j) {
        output << csr_elist[j] << std::endl;
    }
    for (int j = 0; j < m; ++j) {
        output << csr_weight[j] << std::endl;
    }
    output.close();
}

int main(int argc, char **argv) {
    std::string graphdir = argv[1];
    int is_weighted = std::stoi(argv[2]);
    if (is_weighted) {
        Graph g(graphdir + "/origin", true);
        convert2wligra(g, graphdir + "/ligra/weighted.ligra");
    }
    else {
        Graph g(graphdir + "/origin", false);
        convert2ligra(g, graphdir + "/ligra/unweighted.ligra");
    }
    // convert2edgelist(g, graph_dir + "/origin/graph.edgelist");
}