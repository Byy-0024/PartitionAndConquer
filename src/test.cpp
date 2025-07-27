#include "Graph.hpp"
#include "PartitionedGraph.hpp"

#define MERGE

using namespace std;

bool node_pair_less(const pair<node, node> &x, const pair<node, node> &y) {
    if (x.first < y.first) {
        return true;
    }
    else if (x.first == y.first) {
        return x.second < y.second;
    }
    else {
        return false;
    }
}

bool node_pair_greater(const pair<node, node> &x, const pair<node, node> &y) {
    if (x.first > y.first) {
        return true;
    }
    else if (x.first == y.first) {
        return x.second > y.second;
    }
    else {
        return false;
    }
}

template <class T>
class Checker {
	vector<T> expected_answer;
public:
	Checker(vector<T> exp_ans): expected_answer(exp_ans) {}

	bool check(vector<T> answer) {
		assert(answer.size() == expected_answer.size());
		bool is_ok = true;
		int position_wrong = -1;
		for (int i = 0; i < answer.size(); ++i) {
			if (answer.at(i) != expected_answer.at(i)) {
				is_ok = false;
				position_wrong = i;
				break;
			}
		}
		if (is_ok) {
            return true;
		}
		else {
			printf("Something went wrong!\n");
			printf("Answer at %i.\n", position_wrong);
            return false;
		}
	}
};

template <class T>
void read_vector_from_binfile(string filename, vector<T> &v) {
    ifstream input_binfile(filename, ios::binary);
    if (!input_binfile) {
        cerr << "Error: cannot open binfile " << filename << endl;
        exit(1);
    }
    input_binfile.seekg(0, ios::end);
    streamsize size = input_binfile.tellg();
    input_binfile.seekg(0, ios::beg);
    v.resize(size / sizeof(T));
    input_binfile.read(reinterpret_cast<char*>(v.data()), size);
    input_binfile.close();
}

template <class T>
void save_vector_to_binfile(vector<T>&v, string filename) {
    ofstream outfile(filename, ios::binary);
    outfile.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    outfile.close();
}

template <class T>
void print_vector(vector<T>&v) {
    for (auto& x : v) {
        cout << x << endl;
    }
}

void solve_cdlp(string graph_dir, string partition_method, int num_threads, int max_iteration) {
    Graph g(graph_dir + "/origin/csr_vlist.bin", graph_dir + "/origin/csr_elist.bin");
    const int m = g.get_edge_num(), n = g.get_vertex_num();
    vector<int> labels(n, 0);
    for (int i = 0; i < n; i ++) labels[i] = i;
    time_counter _t;
    vector<int> vertex2partition;
    string partition_file = graph_dir + "/partition/" + partition_method + "/P" + to_string(num_threads) + ".bin";
    read_vector_from_binfile<int>(partition_file, vertex2partition);
    Partitioned_Graph G(g, vertex2partition);
    _t.start();
    G.solve_cdlp(labels, max_iteration);
    _t.stop();
    _t.print("parallelizing CDLP with P&P framework");
    printf("Vertex labels: \n");
    for (int i = 0; i < 10; i++) {
        printf("  vertex: %d, label: %d\n", i, labels[i]);
    }
    printf("  ... \n");
}

void solve_jaccard(string graph_dir, string query_file, string partition_method, int num_threads) {
    Graph g(graph_dir + "/origin/csr_vlist.bin", graph_dir + "/origin/csr_elist.bin");
    vector<node> queries;
    read_vector_from_binfile<node>(query_file, queries);
    const int m = g.get_edge_num(), n = g.get_vertex_num();
    int d = queries.size();
    for (int i = 0; i < d; i++) {
        queries[i] %= m;
        queries[i] = g.get_edge(queries[i]).first;
    }
    sort(queries.begin(), queries.end());
    vector<node> tmp;
    tmp.reserve(d);
    tmp.push_back(queries[0]);
    for (size_t i = 1; i < queries.size(); i++) {
        if (queries[i] != queries[i - 1]) tmp.push_back(queries[i]);
    }
    tmp.shrink_to_fit();
    swap(tmp, queries);
    vector<node>().swap(tmp); 
    vector<double> res(d * d, 0);
    vector<int> vertex2partition;
    string partition_file = graph_dir + "/partition/" + partition_method + "/P" + to_string(num_threads) + ".bin";
    read_vector_from_binfile<int>(partition_file, vertex2partition);
    Partitioned_Graph G(g, vertex2partition);
    time_counter _t;
    _t.start();
#ifdef MERGE
    G.solve_jsim_merge(queries, res);
#else
    G.solve_jsim_galloping(queries, res);
#endif
    _t.stop();
    _t.print("parallelizing JSIM with P&P framework");
}

void solve_lcc(string graph_dir, string partition_method, int num_threads) {
    Graph g(graph_dir + "/origin/csr_vlist.bin", graph_dir + "/origin/csr_elist.bin");
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    vector<double> lcc_coeffs(n, 0.0);
    time_counter _t;
    vector<int> vertex2partition;
    string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(num_threads) + ".bin";
    read_vector_from_binfile<int>(partitionfile, vertex2partition);
    Partitioned_Graph G(g, vertex2partition);
    printf("Task information: #Threads = %d, #Vertices = %d, #Queries = %d\n", num_threads, n, m);
    _t.start();
#ifdef MERGE
    G.solve_lcc_merge(lcc_coeffs);
#else
    G.solve_lcc_galloping(lcc_coeffs);
#endif
    _t.stop();
    _t.print("parallelizing LCC with P&P framework");
}

void solve_triangle_counting(string graph_dir, string partition_method, int num_threads) {
    Graph g(graph_dir + "/origin/csr_vlist.bin", graph_dir + "/origin/csr_elist.bin");
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    size_t res = 0;
    time_counter _t;
    vector<int> vertex2partition;
    string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(num_threads) + ".bin";
    read_vector_from_binfile<int>(partitionfile, vertex2partition);
    Partitioned_Graph G(g, vertex2partition);
    G.reorder();
    printf("Task information: #Threads = %d, #Vertices = %d, #Queries = %d\n", num_threads, n, m);
    _t.start();
#ifdef MERGE
    res = G.solve_triangle_counting_merge();
#else
    res = G.solve_triangle_counting_galloping();
#endif
    _t.stop();
    _t.print("parallelizing TC with P&P framework");
    printf("#Triangles: %lu.\n", res); 
}

void solve_page_rank(string graph_dir, string partition_method, int num_threads, float damping_factor, int num_iterations) {
    Graph g(graph_dir + "/origin/csr_vlist.bin", graph_dir + "/origin/csr_elist.bin");
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    vector<float> pr_vals(n, 1.0f);
    time_counter _t;
    vector<int> vertex2partition;
    string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(num_threads) + ".bin";
    read_vector_from_binfile<int>(partitionfile, vertex2partition);
    Partitioned_Graph G(g, vertex2partition);
    G.reorder();
    printf("Task information: #Threads = %d, Damping factor = %.3f, #Iterations = %d\n", num_threads, damping_factor, num_iterations);
    _t.start();
    G.solve_page_rank(pr_vals, damping_factor, num_iterations);
    _t.stop();
    _t.print("partitioned PageRank");
    printf("PageRank values (head 10): \n");
    for (int i = 0; i < 10; ++i) {
        printf("  vertex: %d, pr_vals: %.3f .\n", i, pr_vals[i]);
    }
}

int main(int argc, char** argv) {
    string task = argv[1];
    if (task == "partition") {
        string graphdir = argv[2];
        size_t partition_num = stoul(argv[3]);
        string partition_method = argv[4];
        Graph g(graphdir + "/origin/csr_vlist.bin", graphdir + "/origin/csr_elist.bin");
        vector<int> vertex2partition; 
        time_counter _t;
        _t.start();
        if (partition_method == "URP") g.URP(partition_num, vertex2partition);
        if (partition_method == "GRP") g.GRP(partition_num, vertex2partition);
        if (partition_method == "FGP") g.FGP(partition_num, vertex2partition);
        _t.stop();
        _t.print("partition");
        Partitioned_Graph G(g, vertex2partition);
        G.evaluate_partition();
        string outputdir = graphdir + "/partition/" + partition_method;
        save_vector_to_binfile<int>(vertex2partition, outputdir + "/P" + to_string(partition_num) + ".bin");
    }
    if (task == "generate_queries") {
        size_t query_num = stoul(argv[2]);
        string outputfile = argv[3];
        vector<node> queries;
        for (size_t i = 0; i < query_num; ++i) {
            queries.push_back(rand());
            queries.push_back(rand());
        }
        save_vector_to_binfile<node>(queries, outputfile);
    }
    if (task == "cdlp") {
        string graph_dir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int num_threads = stoi(argv[5]);
        int max_iteration = stoi(argv[6]);
        solve_cdlp(graph_dir, partition_method, num_threads, max_iteration);
    }
    if (task == "jsim") {
        string graph_dir = argv[2];
        string query_file = argv[3];
        string intersection_method = argv[4];
        bool is_partition = stod(argv[5]);
        string partition_method = argv[6];
        int num_threads = stod(argv[7]);
        solve_jaccard(graph_dir, query_file, partition_method, num_threads);
    }
    if (task == "lcc") {
        string graph_dir = argv[2];
        string intersection_method = argv[3];
        bool is_partition = stod(argv[4]);
        string partition_method = argv[5];
        int num_threads = stod(argv[6]);
        solve_lcc(graph_dir, partition_method, num_threads);
    }
    if (task == "tc") {
        string graphdir = argv[2];
        string intersection_method = argv[3];
        bool is_partition = stod(argv[4]);
        string partition_method = argv[5];
        int num_threads = stod(argv[6]);
        solve_triangle_counting(graphdir, partition_method, num_threads); 
    }
    if (task == "pr") {
        string graphdir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int num_threads = stoi(argv[5]);
        float damping_factor = stof(argv[6]);
        int num_iterations = stoi(argv[7]);
        solve_page_rank(graphdir, partition_method, num_threads, damping_factor, num_iterations);
    }
    return 0;
}