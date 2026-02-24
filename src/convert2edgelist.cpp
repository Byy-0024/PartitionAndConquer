# include "Graph.hpp"

void convert2edgelist(const Graph &g, const std::string output_filename) {
    const size_t n = g.get_vertex_num();
    const EID* csr_vlist = g.get_csr_vlist();
    const VID* csr_elist = g.get_csr_elist();
    const Weight* csr_weight = g.get_csr_weight();

    std::ofstream output(output_filename);
    for (VID u = 0; u < n; ++u) {
        for (EID i = csr_vlist[u]; i < csr_vlist[u+1]; ++i) {
            VID v = csr_elist[i];
            Weight w = csr_weight[i];
            output << u << " " << v << " " << w << std::endl;
        }
    }
    output.close();
}

int main(int argc, char **argv) {
    std::string graphdir = argv[1];
    
    Graph g(graphdir + "/origin", true);
    convert2edgelist(g, graphdir + "/edgelist");
}