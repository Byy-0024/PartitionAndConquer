#include "Graph.hpp"

using namespace std;

bool node_pair_less(const pair<VID, VID> &x, const pair<VID, VID> &y) {
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

bool node_pair_greater(const pair<VID, VID> &x, const pair<VID, VID> &y) {
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

void solve_bfs(string graph_dir, bool is_partition, string partition_method, int partition_num, int num_threads) {
    Graph g(graph_dir + "/origin", false);
    const int m = g.get_edge_num(), n = g.get_vertex_num();
    vector<int> parents(n, 0);
    time_counter _t;
    if (is_partition) {
        vector<int> vertex2partition;
        string partition_file = graph_dir + "/partition/" + partition_method + "/P" + to_string(partition_num) + ".bin";
        read_vector_from_binfile<int>(partition_file, vertex2partition);
        Partitioned_Graph G(g, vertex2partition, num_threads);
        // G.reorder();
        _t.start();
        G.solve_bfs(0, parents);
        _t.stop();
        _t.print("partitioned BFS");
    }
    else {
        _t.start();
        g.solve_bfs(0, parents);
        _t.stop();
        _t.print("serial BFS");
    }
    printf("Vertex Rank: \n");
    for (int i = 0; i < 10; i++) {
        printf("  vertex: %d, rank: %d\n", i, parents[i]);
    }
    printf("  ... \n");
}

void solve_cdlp(string graph_dir, bool is_partition, string partition_method, int num_threads, int max_iteration) {
    Graph g(graph_dir + "/origin", false);
    const int m = g.get_edge_num(), n = g.get_vertex_num();
    vector<int> labels(n, 0);
    for (int i = 0; i < n; i ++) labels[i] = i;
    time_counter _t;
    if (is_partition) {
        vector<int> vertex2partition;
        string partition_file = graph_dir + "/partition/" + partition_method + "/P" + to_string(num_threads) + ".bin";
        read_vector_from_binfile<int>(partition_file, vertex2partition);
        Partitioned_Graph G(g, vertex2partition, num_threads);
        _t.start();
        G.solve_cdlp(labels, max_iteration);
        _t.stop();
        _t.print("partitioned CDLP");
    }
    else {
        _t.start();
        g.solve_cdlp(labels, max_iteration, num_threads);
        _t.stop();
        _t.print("parallel CDLP");
    }
    printf("Vertex labels: \n");
    for (int i = 0; i < 10; i++) {
        printf("  vertex: %d, label: %d\n", i, labels[i]);
    }
    printf("  ... \n");
}

void solve_kcore(string graph_dir, bool is_partition, string partition_method, int partition_num, int num_threads) {
    Graph g(graph_dir + "/origin", false);
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    time_counter _t;
    if (is_partition) {
        vector<int> vertex2partition;
        string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(partition_num) + ".bin";
        read_vector_from_binfile<int>(partitionfile, vertex2partition);
        Partitioned_Graph G(g, vertex2partition, num_threads);
        // G.reorder();
        vector<int> res(n, 0);
        _t.start();
        G.solve_kcore(res);
        _t.stop();
        _t.print("partitioned Core Decomposition");
    }
    else {
        _t.start();
        vector<int> res(n, 0);
        g.solve_kcore(res, num_threads);
        _t.stop();
        _t.print("parallel Core Decomposition");
    }  
}

void solve_mis(string graph_dir, bool is_partition, string partition_method, int partition_num, int num_threads) {
    Graph g(graph_dir + "/origin", false);
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    time_counter _t;
    vector<int> res(n, 0);
    if (is_partition) {
        vector<int> vertex2partition;
        string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(partition_num) + ".bin";
        read_vector_from_binfile<int>(partitionfile, vertex2partition);
        // _t.start();
        Partitioned_Graph G(g, vertex2partition, num_threads);
        // _t.stop();
        // _t.print("constructing partitioned graph");
        // _t.clean();
        // G.reorder();
        _t.start();
        G.solve_mis(res);
        _t.stop();
        _t.print("partitioned MIS");
        printf("Size of independent set: %d\n", accumulate(res.begin(), res.end(), 0));
    }
    else {
        _t.start();
        g.solve_mis(res, num_threads);
        _t.stop();
        _t.print("parallel MIS");
        printf("Size of independent set: %d\n", accumulate(res.begin(), res.end(), 0));
    }  
}

void solve_sssp(string graph_dir, bool is_partition, string partition_method, int partition_num, int num_threads) {
    Graph g(graph_dir + "/origin", true);
    const int m = g.get_edge_num(), n = g.get_vertex_num();
    int MAX_SSSP_WEIGHT = INT_MAX / 2;
    vector<int> dist(n, MAX_SSSP_WEIGHT);
    time_counter _t;
    if (is_partition) {
        vector<int> vertex2partition;
        string partition_file = graph_dir + "/partition/" + partition_method + "/P" + to_string(partition_num) + ".bin";
        read_vector_from_binfile<int>(partition_file, vertex2partition);
        Partitioned_Graph G(g, vertex2partition, num_threads);
        // G.reorder();
        _t.start();
        G.solve_sssp(0, dist);
        _t.stop();
        _t.print("partitioned SSSP");
    }
    else {
        _t.start();
        g.solve_sssp(dist, 0);
        _t.stop();
        _t.print("parallel SSSP");
    }
    int max_dist = 0;
    for (VID u = 0; u < n; u++) {
        if (dist[u] == MAX_SSSP_WEIGHT) continue;
        if (dist[u] > max_dist) max_dist = dist[u];
    }
    printf("Max distance: %d\n", max_dist);
    printf("Vertex distances: \n");
    for (int i = 0; i < 10; i++) {
        printf("  vertex: %d, distance: %d\n", i, dist[i]);
    }
    printf("  ... \n");
}

void solve_triangle_counting(string graph_dir, string intersection_method, bool is_partition, string partition_method, int partition_num, int num_threads) {
    Graph g(graph_dir + "/origin", false);
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    size_t res = 0;
    time_counter _t;
    if (is_partition) {
        vector<int> vertex2partition;
        string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(partition_num) + ".bin";
        read_vector_from_binfile<int>(partitionfile, vertex2partition);
        Partitioned_Graph G(g, vertex2partition, num_threads);
        // G.reorder();
        _t.start();
        if (intersection_method == "merge") 
            res = G.solve_triangle_counting_merge();
        else if (intersection_method == "galloping") 
            res = G.solve_triangle_counting_galloping();
        _t.stop();
        _t.print("partitioned " + intersection_method);
    }
    else {
        _t.start();
        if (intersection_method == "merge") 
            res = g.solve_triangle_counting_merge(num_threads);
        else if (intersection_method == "galloping") 
            res = g.solve_triangle_counting_galloping(num_threads);
        _t.stop();
        _t.print("paralle " + intersection_method);
    }
    // printf("#Triangles: %lu.\n", accumulate(res.begin(), res.end(), (size_t) 0)); 
    printf("#Triangles: %lu.\n", res); 
}

void solve_page_rank(string graph_dir, bool is_partition, string partition_method, int partition_num, int num_threads, float damping_factor, int num_iterations) {
    Graph g(graph_dir + "/origin", false);
    const int n = g.get_vertex_num(), m = g.get_edge_num();
    vector<float> pr_vals(n, 1.0f);
    time_counter _t;
    if (is_partition) {
        vector<int> vertex2partition;
        string partitionfile = graph_dir + "/partition/" + partition_method + "/P" + to_string(partition_num) + ".bin";
        read_vector_from_binfile<int>(partitionfile, vertex2partition);
        Partitioned_Graph G(g, vertex2partition, num_threads);
        G.reorder();
        printf("Task information: #Threads = %d, Damping factor = %.3f, #Iterations = %d\n", num_threads, damping_factor, num_iterations);
        _t.start();
        // G.page_rank_residual(pr_vals, damping_factor, num_iterations);
        G.page_rank(pr_vals, damping_factor, num_iterations);
        _t.stop();
        _t.print("partitioned PageRank");
    }
    else {
        printf("Task information: #Threads = %d, Damping factor = %.3f, #Iterations = %d\n", num_threads, damping_factor, num_iterations);
        _t.start();
        g.page_rank(pr_vals, damping_factor, num_iterations, num_threads);
        _t.stop();
        _t.print("parallel PageRank");
    }
    float max_pr_rank = *max_element(pr_vals.begin(), pr_vals.end());
    float min_pr_rank = *min_element(pr_vals.begin(), pr_vals.end());
    printf("Max PageRank: %f, Min PageRank: %f\n", max_pr_rank, min_pr_rank);
    printf("PageRank values (head 10): \n");
    for (int i = 0; i < 10; ++i) {
        printf("  vertex: %d, pr_vals: %f .\n", i, pr_vals[i]);
    }
}

int main(int argc, char** argv) {
    string task = argv[1];
    if (task == "partition") {
        string graphdir = argv[2];
        size_t partition_num = stoul(argv[3]);
        string outputdir = argv[4];
        Graph g(graphdir + "/origin", false);
        vector<int> vertex2partition; 
        time_counter _t;
        _t.start();
        // g.greedy_partition_2(partition_num, vertex2partition);
        g.greedy_partition(partition_num, vertex2partition);
        // g.random_partition(partition_num, vertex2partition);
        _t.stop();
        // _t.print("random partition");
        _t.print("greedy partition");
        Partitioned_Graph G(g, vertex2partition, partition_num);
        G.evaluate_partition();
        save_vector_to_binfile<int>(vertex2partition, outputdir + "/P" + to_string(partition_num) + ".bin");
    }
    if (task == "generate_queries") {
        size_t query_num = stoul(argv[2]);
        string outputfile = argv[3];
        vector<VID> queries;
        for (size_t i = 0; i < query_num; ++i) {
            queries.push_back(rand());
            queries.push_back(rand());
        }
        save_vector_to_binfile<VID>(queries, outputfile);
    }
    if (task == "bfs") {
        string graph_dir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int partition_num = stoi(argv[5]);
        int num_threads = stoi(argv[6]);
        solve_bfs(graph_dir, is_partition, partition_method, partition_num, num_threads);
    }
    if (task == "cdlp") {
        string graph_dir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int num_threads = stoi(argv[5]);
        int max_iteration = stoi(argv[6]);
        solve_cdlp(graph_dir, is_partition, partition_method, num_threads, max_iteration);
    }
    if (task == "kcore") {
        string graphdir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int partition_num = stod(argv[5]);
        int num_threads = stod(argv[6]);
        solve_kcore(graphdir, is_partition, partition_method, partition_num, num_threads); 
    }
    if (task == "mis") {
        string graphdir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int partition_num = stod(argv[5]);
        int num_threads = stod(argv[6]);
        solve_mis(graphdir, is_partition, partition_method, partition_num, num_threads); 
    }
    if (task == "sssp") {
        string graph_dir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int partition_num = stod(argv[5]);
        int num_threads = stoi(argv[6]);
        solve_sssp(graph_dir, is_partition, partition_method, partition_num, num_threads);
    }
    if (task == "tc") {
        string graphdir = argv[2];
        string intersection_method = argv[3];
        bool is_partition = stod(argv[4]);
        string partition_method = argv[5];
        int partition_num = stod(argv[6]);
        int num_threads = stod(argv[7]);
        solve_triangle_counting(graphdir, intersection_method, is_partition, partition_method, partition_num, num_threads); 
    }
    if (task == "pr") {
        string graphdir = argv[2];
        bool is_partition = stod(argv[3]);
        string partition_method = argv[4];
        int partition_num = stoi(argv[5]);
        int num_threads = stoi(argv[6]);
        float damping_factor = stof(argv[7]);
        int num_iterations = stoi(argv[8]);
        solve_page_rank(graphdir, is_partition, partition_method, partition_num, num_threads, damping_factor, num_iterations);
    }
    return 0;
}