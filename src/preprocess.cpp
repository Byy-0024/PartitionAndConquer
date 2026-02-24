#include "Graph.hpp"

using namespace std;

void convert2binfile(string graphfile, string outputdir) {
    Graph g(graphfile);
    g.save_as_csr_binfile(outputdir + "/csr_vlist.bin", outputdir + "/csr_elist.bin");
}

void add_weight(string csrdir) {
    Graph g(csrdir, false);
    const VID n = g.get_vertex_num();
    const EID m = g.get_edge_num();
    const Weight max_weight = log2(n);

    const EID* csr_vlist = g.get_csr_vlist();
    const VID* csr_elist = g.get_csr_elist();

    vector<int> weight(m, 0);
    for (VID u = 0; u < n; u++) {
        for (EID i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
            VID v = csr_elist[i];
            if (u < v) {
                weight[i] = rand() % max_weight + 1;
            }
            else {
                EID _lb = lower_bound(csr_elist + csr_vlist[v], csr_elist + csr_vlist[v+1], u) - csr_elist;
                weight[i] = weight[_lb];
            }
        }
    }

    ofstream output_csr_weight_binfile(csrdir + "/csr_weight.bin", ios::binary);
    output_csr_weight_binfile.write(reinterpret_cast<const char*>(weight.data()), sizeof(Weight) * weight.size());
    printf("Size of csr_weight: %lu.\n", weight.size());
    output_csr_weight_binfile.close();
}

int main(int argc, char **argv) {
    string graphdir = argv[1];
    string graphfile = argv[2];
    string csrdir = graphdir + "/origin";
    convert2binfile(graphdir + graphfile, csrdir);
    add_weight(csrdir);
    return 0;
}