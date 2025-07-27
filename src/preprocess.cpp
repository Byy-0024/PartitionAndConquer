#include "Graph.hpp"

using namespace std;

void convert2binfile(string graphfile, string outputdir) {
    // Graph g(graphfile);
    // g.save_as_csr_binfile(outputdir + "/origin/csr_vlist.bin", outputdir + "/origin/csr_elist.bin");
    Graph g(graphfile + "/origin/csr_vlist.bin", graphfile + "/origin/csr_elist.bin");
    g.save_as_ligra(outputdir + "adjgraph");
}

int main(int argc, char **argv) {
    string graphfile = argv[1];
    string outputdir = argv[2];
    convert2binfile(graphfile, outputdir);
    return 0;
}