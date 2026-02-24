#include "Graph.hpp"

using namespace std;

void time_counter::start() {
    // t_start = chrono::steady_clock::now();
    t_start = chrono::high_resolution_clock::now();
}

void time_counter::stop() {
    // t_end = chrono::steady_clock::now();
    t_end = chrono::high_resolution_clock::now();
    t_cnt += chrono::duration_cast<chrono::nanoseconds>(t_end - t_start).count();
}

void time_counter::print(string s) {
    if (t_cnt < 1000) printf("Time used for %s: %llu ns.\n", s.c_str(), t_cnt);
    else if (t_cnt < 1000000) printf("Time used for %s: %.3f us.\n", s.c_str(), (float) t_cnt / 1000);
    else if (t_cnt < (ull) 1000000000) printf("Time used for %s: %.3f ms.\n", s.c_str(), (float) t_cnt / 1000000);
    else printf("Time used for %s: %.3f s.\n", s.c_str(), (float) t_cnt / (float) 1000000000);
}

void time_counter::clean() {
    t_cnt = 0;
}

Graph::Graph(const string graphfile) {
    bool has_zero = false;
	vertex_num = 0;
	edge_num = 0;
	FILE *fp = fopen(graphfile.c_str(), "r");
	if (fp == NULL) {
        std::cout << "fail to open " << graphfile << std::endl;
    }
	vector<pair<VID, VID>> edge_list;

    char line[512];
    while (fgets(line, 512, fp) != NULL) {
        if (line[0] == '#' || line[0] == '%') continue;
        int u = 0, v = 0;
        const char *c = line;
        while (isdigit(*c))
            u = (u << 1) + (u << 3) + (*c++ - 48);
        c++;
        while (isdigit(*c))
            v = (v << 1) + (v << 3) + (*c++ - 48);
		if (u == 0 || v == 0) {
			has_zero = true;
		}
		if (u == v) continue;
		if (u > vertex_num) vertex_num = u;
		if (v > vertex_num) vertex_num = v;
        edge_list.push_back({(VID) u, (VID) v});
    }    
    fclose(fp);

	if (has_zero) vertex_num++;
    printf("Vertex num: %lu.\n", vertex_num);
    vector<vector<VID>> adj_list(vertex_num);  
	for (auto e : edge_list) {
		if (has_zero) {
			adj_list[e.first].push_back(e.second);
			adj_list[e.second].push_back(e.first);
		}
		else {
			adj_list[e.first-1].push_back(e.second-1);
			adj_list[e.second-1].push_back(e.first-1);
		}
	}
    vector<pair<VID, VID>>().swap(edge_list);
	for (VID i = 0; i < vertex_num; ++i){
		if (!adj_list[i].empty()) {
			sort(adj_list[i].begin(), adj_list[i].end());
			adj_list[i].erase(unique(adj_list[i].begin(), adj_list[i].end()), adj_list[i].end());
			edge_num += adj_list[i].size();
		} 
	}
	
    csr_vlist.reserve(vertex_num + 1);
    csr_elist.reserve(edge_num);
    for (size_t i = 0; i < vertex_num; i++) {
        csr_vlist.push_back(csr_elist.size());
        csr_elist.insert(csr_elist.end(), adj_list[i].begin(), adj_list[i].end());
    }
    csr_vlist.push_back(edge_num);
    vector<vector<VID>>().swap(adj_list);

	printf("Number of nodes: %lu, number of edges: %lu, size of adj-list: %.3f MB!\n", 
			vertex_num, edge_num, (float)(edge_num * sizeof(VID) + vertex_num * sizeof(EID)) / (1<<20));
}

Graph::Graph(const string csrdir, bool is_weighted) {
    string csr_vlist_binfile = csrdir + "/csr_vlist.bin";
    string csr_elist_binfile = csrdir + "/csr_elist.bin";
    string csr_weight_binfile = csrdir + "/csr_weight.bin";
    ifstream input_csr_vlist(csr_vlist_binfile, ios::binary);
    if (!input_csr_vlist.is_open()) {
        cerr << "Unable to open csr vlist binfile from " << csr_vlist_binfile << endl;
        return;
    }
    input_csr_vlist.seekg(0, ios::end);
    streamsize size = input_csr_vlist.tellg();
    input_csr_vlist.seekg(0, ios::beg);
    csr_vlist.resize(size / sizeof(EID));
    input_csr_vlist.read(reinterpret_cast<char*>(csr_vlist.data()), size);
    input_csr_vlist.close();
    vertex_num = csr_vlist.size() - 1;

    ifstream input_csr_elist(csr_elist_binfile, ios::binary);
    if (!input_csr_elist.is_open()) {
        cerr << "Unable to open csr elist binfile from " << csr_elist_binfile << endl;
        return;
    }
    input_csr_elist.seekg(0, ios::end);
    size = input_csr_elist.tellg();
    input_csr_elist.seekg(0, ios::beg);
    csr_elist.resize(size / sizeof(VID));
    input_csr_elist.read(reinterpret_cast<char*>(csr_elist.data()), size);
    input_csr_elist.close();
    edge_num = csr_elist.size();

    if (is_weighted) {
        ifstream input_csr_weight(csr_weight_binfile, ios::binary);
        if (!input_csr_weight.is_open()) {
            cerr << "Unable to open csr weight binfile from " << csr_weight_binfile << endl;
            return;
        }
        input_csr_weight.seekg(0, ios::end);
        size = input_csr_weight.tellg();
        input_csr_weight.seekg(0, ios::beg);
        csr_weight.resize(size / sizeof(Weight));
        input_csr_weight.read(reinterpret_cast<char*>(csr_weight.data()), size);
        input_csr_weight.close();
    }
    
    printf("Has read graph! #Vertices: %lu, #Edges: %lu, #Weights: %lu.\n", vertex_num, edge_num, csr_weight.size());
}

Graph::Graph(const vector<vector<VID>> &adj_list) {
    vertex_num = adj_list.size();
    csr_vlist.reserve(vertex_num + 1);
    csr_vlist.push_back(0);
    for (auto neis: adj_list) {
        csr_elist.insert(csr_elist.end(), neis.begin(), neis.end());
        csr_vlist.push_back(csr_elist.size());
    }
    edge_num = csr_elist.size();
    printf("Has constructed graph! #Vertices: %lu, #Edges: %lu.\n", vertex_num, edge_num);
}

const EID* Graph::get_csr_vlist() const {
    return csr_vlist.data();
}

const VID* Graph::get_csr_elist() const {
    return csr_elist.data();
}

const Weight* Graph::get_csr_weight() const {
    return csr_weight.data();
}

bool Graph::has_edge(VID u, VID v) const {
    return get_deg(u) < get_deg(v) ? has_directed_edge(u, v) : has_directed_edge(v, u);
}

bool Graph::has_directed_edge(VID u, VID v) const {
    size_t ptr = lower_bound(csr_elist.begin() + csr_vlist[u], csr_elist.begin() + csr_vlist[u+1], v) - csr_elist.begin();
    return ptr < csr_vlist[u+1] && csr_elist[ptr] == v;
}

bool Graph::is_weighted() const {
    return !csr_weight.empty();
}

VID Graph::get_nei(VID u, EID offset) const {
    return csr_elist[csr_vlist[u] + offset];
}

pair<VID,VID> Graph::get_edge(EID offset) const {
    VID v = csr_elist[offset];
    VID u = upper_bound(csr_vlist.begin(), csr_vlist.end(), offset) - csr_vlist.begin() - 1;
    return {u, v};
}

size_t Graph::cnt_common_neis_galloping(VID u, VID v) const {
    EID u_start_ptr = csr_vlist[u], u_end_ptr = csr_vlist[u+1], v_start_ptr = csr_vlist[v], v_end_ptr = csr_vlist[v+1], res = 0;
    if (u_end_ptr - u_start_ptr > v_end_ptr - v_start_ptr) {
        swap(u_start_ptr, v_start_ptr);
        swap(u_end_ptr, v_end_ptr);
    }
    for (EID i = u_start_ptr; i < u_end_ptr; i ++) {
        VID target = csr_elist[i];
        EID offset = lower_bound(csr_elist.begin() + v_start_ptr, csr_elist.begin() + v_end_ptr, target) - csr_elist.begin();
        if (offset < v_end_ptr && csr_elist[offset] == target) res ++;
    }
    return res;
}

size_t Graph::cnt_common_neis_galloping(const vector<VID> &q) const {
    size_t res = 0, min_deg = vertex_num;
    VID u;
    for (VID v: q) {
        if (get_deg(v) < min_deg) {
            min_deg = get_deg(v);
            u = v;
        }
    }
    size_t u_start_ptr = csr_vlist[u], u_end_ptr = csr_vlist[u+1];
    for (size_t i = u_start_ptr; i < u_end_ptr; i ++) {
        VID target = csr_elist[i];
        bool found = true;
        for (VID v: q) {
            if (u == v) continue;
            size_t v_start_ptr = csr_vlist[v], v_end_ptr = csr_vlist[v+1];
            size_t offset = lower_bound(csr_elist.begin() + v_start_ptr, csr_elist.begin() + v_end_ptr, target) - csr_elist.begin();
            if (offset >= v_end_ptr || csr_elist[offset] != target) {
                found = false;
                break;
            }
        }
        if (found) res ++;
    }
    return res;
}

size_t Graph::cnt_common_neis_merge(VID u, VID v) const {
    size_t u_ptr = csr_vlist[u], v_ptr = csr_vlist[v], res = 0;
    while (u_ptr < csr_vlist[u+1] && v_ptr < csr_vlist[v+1]) {
        if (csr_elist[u_ptr] == csr_elist[v_ptr]) {
            res++;
            u_ptr++;
            v_ptr++;
        }
        else if (csr_elist[u_ptr] < csr_elist[v_ptr]) u_ptr++;
        else v_ptr++;
    }
    return res;
}

size_t Graph::cnt_common_neis_merge(VID u, const vector<VID> &nodes) const {
    size_t u_ptr = csr_vlist[u], v_ptr = 0, res = 0;
    while (u_ptr < csr_vlist[u+1] && v_ptr < nodes.size()) {
        if (csr_elist[u_ptr] == nodes[v_ptr]) {
            res++;
            u_ptr++;
            v_ptr++;
        }
        else if (csr_elist[u_ptr] < nodes[v_ptr]) u_ptr++;
        else v_ptr++;
    }
    return res;
}

size_t Graph::cnt_common_neis_merge(const vector<VID> &q) const {
    vector<VID> tmp_res1, tmp_res2;
    size_t res_upper_bound = get_deg(q[0]);
    tmp_res1.reserve(res_upper_bound);
    tmp_res2.reserve(res_upper_bound);
    get_neis(q[0], tmp_res1);
    for (size_t i = 1; i < q.size()-1; i ++) {
        get_common_neis(q[i], tmp_res1, tmp_res2);
        swap(tmp_res1, tmp_res2);
        tmp_res2.clear();
    }
    size_t res = cnt_common_neis_merge(q[q.size()-1], tmp_res1);
    vector<VID>().swap(tmp_res1);
    vector<VID>().swap(tmp_res2);
    return res;
}

size_t Graph::cnt_common_neis_less_than_merge(VID u, VID v, VID threshold) const {
    size_t u_ptr = csr_vlist[u], v_ptr = csr_vlist[v], res = 0;
    while (u_ptr < csr_vlist[u+1] && v_ptr < csr_vlist[v+1] 
                && csr_elist[u_ptr] < threshold && csr_elist[v_ptr] < threshold) {
        if (csr_elist[u_ptr] == csr_elist[v_ptr]) {
            res++;
            u_ptr++;
            v_ptr++;
        }
        else if (csr_elist[u_ptr] < csr_elist[v_ptr]) u_ptr++;
        else v_ptr++;
    }
    return res;
}


EID Graph::get_edge_num() const {
    return edge_num;
}

VID Graph::get_vertex_num() const {
    return vertex_num;
}

size_t Graph::get_deg(VID u) const {
    return csr_vlist[u+1] - csr_vlist[u];
}

size_t Graph::get_max_deg() const {
    size_t res = 0;
    omp_set_num_threads(32);
#pragma omp parallel for reduction(max:res)
    for (size_t i = 0; i < vertex_num; i ++) 
        res = max(res, get_deg(i));
    return res;
}

size_t Graph::solve_friend_triangle_galloping(const VID root, const vector<VID> &nodes, const int num_threads) const {
    size_t res = 0;
    omp_set_num_threads(num_threads);
#pragma omp parallel for reduction(+:res)
    for (size_t i = 0; i < nodes.size(); i ++) {
        if (nodes[i] == root) continue;
#pragma omp parallel for reduction(+:res)
        for (size_t j = i+1; j < nodes.size(); j ++) {
            if (nodes[j] == root) continue;
            size_t local_cnt = cnt_common_neis_galloping({root, nodes[i], nodes[j]});
// #pragma omp atomic
            res += local_cnt;
        }
    }
    return res;
}

size_t Graph::solve_friend_triangle_merge(const VID root, const vector<VID> &nodes, const int num_threads) const {
    size_t res = 0;
    omp_set_num_threads(num_threads);
#pragma omp parallel for reduction(+:res)
    for (size_t i = 0; i < nodes.size(); i ++) {
        if (nodes[i] == root) continue;
        for (size_t j = i+1; j < nodes.size(); j ++) {
            if (nodes[j] == root) continue;
            size_t local_cnt = cnt_common_neis_merge({root, nodes[i], nodes[j]});
// #pragma omp atomic
            res += local_cnt;
        }
    }
    return res;
}

size_t Graph::solve_triangle_counting_galloping(const int num_threads) const {
    size_t res = 0;
#pragma omp parallel for num_threads(num_threads) reduction(+:res) 
    for (VID u = 0; u < vertex_num; u++) {
        for (int j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
            VID v = csr_elist[j];
            size_t local_cnt = cnt_common_neis_galloping(u, v);
            res += local_cnt;
        }
    }
    return res;
}

size_t Graph::solve_triangle_counting_merge(const int num_threads) const {
    size_t res = 0;
#pragma omp parallel for num_threads(num_threads) reduction(+:res) 
    for (VID u = 0; u < vertex_num; u++) {
        for (int j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
            VID v = csr_elist[j];
            if (u <= v) continue;
            // size_t local_cnt = cnt_common_neis_merge(u, v);
            size_t local_cnt = cnt_common_neis_less_than_merge(u,v,v);
            res += local_cnt;
        }
    }
    return res;
}

// size_t Graph::solve_triangle_counting_galloping(const int inner_threads, const int outer_threads) const {
//     size_t res = 0;
//     const int total_threads = inner_threads * outer_threads;
// #pragma omp parallel for num_threads(total_threads) reduction(+:res)
//     for (size_t idx = 0; idx < csr_vlist[vertex_num]; idx++) {
//         VID u = lower_bound(csr_vlist.begin(), csr_vlist.end(), idx) - csr_vlist.begin();
//         if (csr_vlist[u] > idx) u--;
//         VID v = csr_elist[idx];
//         size_t local_cnt = cnt_common_neis_galloping(u, v);
//         res += local_cnt;
//     }

//     return res;
// // #pragma omp parallel for reduction(+:res) num_threads(outer_threads)
// //     for (VID u = 0; u < vertex_num; u++) {
// // #pragma omp parallel for reduction(+:res) num_threads(inner_threads)
// //         for (int j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
// //             VID v = csr_elist[j];
// //             size_t local_cnt = cnt_common_neis_galloping(u, v);
// //             res += local_cnt;
// //         }
// //     }
// //     return res;
// }

// size_t Graph::solve_triangle_counting_merge(const int inner_threads, const int outer_threads) const {
//     size_t res = 0;
// #pragma omp parallel for num_threads(outer_threads)reduction(+:res) 
//     for (VID u = 0; u < vertex_num; u++) {
// #pragma omp parallel for num_threads(inner_threads) reduction(+:res)
//         for (int j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
//             VID v = csr_elist[j];
//             size_t local_cnt = cnt_common_neis_merge(u, v);
//             res += local_cnt;
//         }
//     }
//     return res;
// }

ull Graph::size_in_bytes() const {
    return csr_vlist.size() * sizeof(int) + csr_elist.size() * sizeof(VID);
}

void Graph::batch_cnt_common_neis_galloping(const vector<pair<VID, VID>> &queries, vector<size_t> &res, int num_threads) const {
    const int N = queries.size();
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (size_t i = 0; i < N; i ++) {
            res[i] = cnt_common_neis_galloping(queries[i].first, queries[i].second);
    }
}

void Graph::batch_cnt_common_neis_galloping(const vector<vector<VID>> &queries, vector<size_t> &res, int num_threads) const {
    const int N = queries.size();
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (size_t i = 0; i < N; i ++) {
            res[i] = cnt_common_neis_galloping(queries[i]);
    }
}

void Graph::batch_cnt_common_neis_merge(const vector<pair<VID, VID>> &queries, vector<size_t> &res, int num_threads) const {
    const int N = queries.size();
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (size_t i = 0; i < N; i ++) {
            res[i] = cnt_common_neis_merge(queries[i].first, queries[i].second);
    }
}

void Graph::batch_cnt_common_neis_merge(const vector<vector<VID>> &queries, vector<size_t> &res, int num_threads) const {
    const int N = queries.size();
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (size_t i = 0; i < N; i ++) {
            res[i] = cnt_common_neis_merge(queries[i]);
    }
}

void Graph::bfs(VID source, vector<VID> &parents) const {
    parents.assign(vertex_num, -1);
    parents[source] = INT_MAX;
    queue<VID> frontier;
    frontier.push(source);

    while (!frontier.empty()) {
        VID u = frontier.front();
        frontier.pop();
        for (int j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
            VID v = csr_elist[j];
            if (parents[v] == -1) {
                parents[v] = u;
                frontier.push(v);
            }
        }
    }
}

void Graph::get_common_neis(VID u, VID v, vector<VID>& res) const {
	int u_ind = csr_vlist[u], v_ind = csr_vlist[v];
	while(u_ind < csr_vlist[u+1] && v_ind < csr_vlist[v+1]) {
		if (csr_elist[u_ind] == csr_elist[v_ind]) {
			res.emplace_back(csr_elist[u_ind]);
			u_ind++;
			v_ind++;
		}
		else if (csr_elist[u_ind] < csr_elist[v_ind]) u_ind++;
		else v_ind++;
	}
}

void Graph::get_common_neis(VID u, const vector<VID> &nodes, vector<VID> &res) const {
	res.clear();
	int u_ind = csr_vlist[u], v_ind = 0;
	while (u_ind < csr_vlist[u+1] && v_ind < nodes.size()) {
		if (csr_elist[u_ind] == nodes[v_ind]) {
			res.push_back(nodes[v_ind]);
			u_ind++;
			v_ind++;
		}
		else if (csr_elist[u_ind] < nodes[v_ind]) u_ind++;
		else v_ind++;
	}
}

void Graph::get_degs(vector<int> &degs) const {
    degs.clear();
    degs.resize(vertex_num);
    for (VID u = 0; u < vertex_num; u++) {
        degs[u] = csr_vlist[u+1] - csr_vlist[u];
    }
}

void Graph::get_neis(VID u, vector<VID> &neis) const {
    neis.clear();
    if (csr_vlist[u] == csr_vlist[u+1]) return;
    neis.reserve(csr_vlist[u+1] - csr_vlist[u]);
    neis.insert(neis.end(), csr_elist.begin() + csr_vlist[u], csr_elist.begin() + csr_vlist[u+1]);
}

void Graph::get_two_hop_neis(VID u, vector<VID> &res) const {
    vector<bool> seen(vertex_num, false);
    res.clear();
    res.reserve(vertex_num);
    size_t u_start_ptr = csr_vlist[u], u_end_ptr = csr_vlist[u+1];
    for (size_t i = u_start_ptr; i < u_end_ptr; i++) {
        VID v = csr_elist[i];
        size_t v_start_ptr = csr_vlist[v], v_end_ptr = csr_vlist[v+1];
        for (size_t j = v_start_ptr; j < v_end_ptr; j++) {
            VID w = csr_elist[j];
            if (!seen[w]) {
                res.push_back(w);
                seen[w] = true;
            }
        }
    }
    res.shrink_to_fit();
}

// void Graph::local_cluster_coefficient(const vector<VID> &nodes, vector<float> &res) const {
//     res.resize(nodes.size());
//     vector<pair<VID,VID>> queries();
// }

void Graph::page_rank(vector<float> &pr_vals, float damping_factor, int num_iterations, int num_threads) const {
    vector<float> pr_vals_new(vertex_num);
    for (int i = 0; i < num_iterations; i++) {
#pragma omp parallel for num_threads(num_threads)
        for (VID u = 0; u < vertex_num; u++) {
            pr_vals_new[u] = 0;
            pr_vals[u] /= get_deg(u);
        }
#pragma omp parallel for num_threads(num_threads)
        for (VID u = 0; u < vertex_num; u++) {
            for (size_t j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
                VID v = csr_elist[j];
                pr_vals_new[u] += pr_vals[v];
            }
        }
#pragma omp parallel for num_threads(num_threads)
        for (VID u = 0; u < vertex_num; u++) {
            pr_vals[u] = (1 - damping_factor) / vertex_num + damping_factor * pr_vals_new[u];
        }
    }
}

void Graph::print_neis(VID u) const {
    printf("N(%d):", u);
    for (int i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
        printf(" %d", csr_elist[i]);
    }
    printf(".\n");
}

void Graph::random_partition(const size_t part_num, vector<VID> &vertex2partition) {
    vertex2partition.clear();
    vertex2partition.resize(vertex_num);
    for (VID u = 0; u < vertex_num; u++) {
        vertex2partition[u] = rand() % part_num;
    }
}

void Graph::reorder(const vector<VID> &origin_to_new){
	vector<bool> seen(vertex_num, false);
	vector<vector<VID>> adj_list(vertex_num);
	for(size_t i = 0; i < vertex_num; ++i) {
		VID new_id = origin_to_new[i];
		if (new_id == UINT_MAX) continue;
		if (seen[new_id]) printf("Error on VID %lu! Already has VID with label %d!\n", i, new_id);
		else seen[new_id] = true;
        for (size_t j = csr_vlist[i]; j < csr_vlist[i+1]; ++j) {
            VID v = csr_elist[j];
            adj_list[new_id].push_back(origin_to_new[v]);
        }
		if (!adj_list[new_id].empty()) sort(adj_list[new_id].begin(), adj_list[new_id].end());
	}
    csr_vlist.clear();
    csr_elist.clear();
    for (size_t i = 0; i < vertex_num; i++) {
        csr_vlist.push_back(csr_elist.size());
        csr_elist.insert(csr_elist.end(), adj_list[i].begin(), adj_list[i].end());
    }
    csr_vlist.push_back(csr_elist.size());
}

void Graph::save_as_csr_binfile(const string csr_vlist_binfile, const string csr_elist_binfile) {
    ofstream output_csr_vlist_binfile(csr_vlist_binfile, ios::binary);
    output_csr_vlist_binfile.write(reinterpret_cast<const char*>(csr_vlist.data()), sizeof(EID) * csr_vlist.size());
    printf("Size of csr_vlist: %lu.\n", csr_vlist.size());
    output_csr_vlist_binfile.close();

    ofstream output_csr_elist_binfile(csr_elist_binfile, ios::binary);
    output_csr_elist_binfile.write(reinterpret_cast<const char*>(csr_elist.data()), sizeof(VID) * csr_elist.size());
    printf("Size of csr_elist: %lu.\n", csr_elist.size());
    output_csr_elist_binfile.close();
}

void Graph::save_as_ligra(const string ligrafile) const {
    ofstream output_ligrafile(ligrafile);
    output_ligrafile << "AdjacencyGraph" << endl;
    output_ligrafile << vertex_num << endl;
    output_ligrafile << edge_num << endl;
    for (VID u = 0; u < vertex_num; u++) {
        output_ligrafile << csr_vlist[u] << endl;
    }
    for (size_t i = 0; i < csr_elist.size(); i++) {
        output_ligrafile << csr_elist[i] << endl;
    }
    output_ligrafile.close();
}

void Graph::solve_bfs(VID u, vector<VID> &parents) const {
    parents.clear();
    parents.resize(vertex_num, INT_MAX);
    parents[u] = -1;

    vector<VID> frontiers;
    frontiers.reserve(vertex_num);
    frontiers.push_back(u);

    size_t curr_seen_cnt = 0;
    size_t next_seen_cnt = frontiers.size();
    while (next_seen_cnt > curr_seen_cnt) {
        for (size_t i = curr_seen_cnt; i < next_seen_cnt; i++) {
            VID curr = frontiers[i];
            for (size_t j = csr_vlist[curr]; j < csr_vlist[curr + 1]; j++) {
                VID v = csr_elist[j];
                if (parents[v] == INT_MAX) {
                    parents[v] = curr;
                    frontiers.push_back(v);
                }
            }
        }
        curr_seen_cnt = next_seen_cnt;
        next_seen_cnt = frontiers.size();
    }
}

void Graph::solve_cdlp(vector<int> &labels, int max_iterations, int num_threads) const {
    for (int iter = 0; iter < max_iterations; iter++) {
        vector<int> tmp_labels(vertex_num, 0);
#pragma omp parallel for num_threads(num_threads)
        for (VID u = 0; u < vertex_num; u++) {
            if (csr_vlist[u] == csr_vlist[u+1]) continue;
            for (int j = csr_vlist[u]; j < csr_vlist[u+1]; j++) 
                tmp_labels[u] = max(tmp_labels[u], labels[csr_elist[j]]);
        }
        // if (tmp_labels == labels) break;
        labels.swap(tmp_labels);
    }
}

void Graph::solve_kcore(vector<int> &core_numbers, int num_threads) const {
    vector<VID> frontiers;
    vector<VID> to_removes;
    vector<VID> remainings;
   
    frontiers.reserve(vertex_num);
    to_removes.reserve(vertex_num);
    remainings.reserve(vertex_num);
    for (VID u = 0; u < vertex_num; u++) {
        frontiers.push_back(u);
    }

    vector<int> degrees(vertex_num, 0);
    core_numbers.resize(vertex_num, 0);
    for (VID u = 0; u < vertex_num; u++) {
        degrees[u] = csr_vlist[u + 1] - csr_vlist[u];
    }

    size_t largest_core = 0;
    bool is_inner_loop_finished = false;
    bool is_outer_loop_finished = false;
    for (size_t k = 1; k <vertex_num; k++) {
        is_outer_loop_finished = true;
        is_inner_loop_finished = false;
        while (!is_inner_loop_finished) {
            is_inner_loop_finished = true;    
            to_removes.clear();
            remainings.clear();
            for (VID u : frontiers) {
                if (degrees[u] < k) {
                    core_numbers[u] = k-1;
                    degrees[u] = 0;
                    to_removes.push_back(u);
                }
                else {
                    remainings.push_back(u);
                }
            }
            if (!remainings.empty()) is_outer_loop_finished = false;
            if (!to_removes.empty()) is_inner_loop_finished = false;             

            swap(frontiers,remainings);  

            if (is_inner_loop_finished) break;
            for (VID u : to_removes) {
                for (size_t j = csr_vlist[u]; j < csr_vlist[u + 1]; j++) {
                    VID v = csr_elist[j];
                    degrees[v]--;
                }
            }
        }
        if (is_outer_loop_finished) {
            largest_core = k-1;
            break;
        }
    }
    printf("largest core: %lu.\n", largest_core);
}

void Graph::solve_mis(vector<int> &res, int thread_num) const {
    enum {UNDICEDED, CONDITIONALLY_IN, OUT, IN};
    vector<int> flags(vertex_num, CONDITIONALLY_IN);
    size_t round = 0;
    vector<VID> frontier;
    vector<VID> next_frontier;
    frontier.reserve(vertex_num);
    next_frontier.reserve(vertex_num);

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        for (size_t j = csr_vlist[u]; j < csr_vlist[u + 1]; j++) {
            VID v = csr_elist[j];
            if (flags[v] == IN) {if (flags[u] != OUT) flags[u] = OUT;}
            else if (v < u && flags[u] == CONDITIONALLY_IN && flags[v] < OUT) {
                flags[u] = UNDICEDED;
            }
        } 
    }

// #pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        if (flags[u] == CONDITIONALLY_IN) {flags[u] = IN;}
        else if (flags[u] == UNDICEDED) {
            flags[u] = CONDITIONALLY_IN;
            frontier.push_back(u);
        } 
    }

    while (!frontier.empty()) {
        round++;
#pragma omp parallel for num_threads(thread_num)
        for (auto u : frontier) {
            if (flags[u] == OUT) continue;
            for (size_t j = csr_vlist[u]; j < csr_vlist[u + 1]; j++) {
                VID v = csr_elist[j];
                if (flags[v] == IN) {if (flags[u] != OUT) flags[u] = OUT; break;}
                else if (v < u && flags[u] == CONDITIONALLY_IN && flags[v] < OUT) {
                    flags[u] = UNDICEDED;
                }
            }
        }
// #pragma omp parallel for num_threads(thread_num)
        for (auto u : frontier) {
            if (flags[u] == CONDITIONALLY_IN) {flags[u] = IN;}
            else if (flags[u] == UNDICEDED) {
                flags[u] = CONDITIONALLY_IN;
                next_frontier.push_back(u);
            } 
        }
        swap(frontier, next_frontier);
        next_frontier.clear();
    }

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        res[u] = (flags[u] == IN);
    }
    printf("round: %lu\n", round); 
}

void Graph::solve_jsim_galloping(const std::vector<VID> &queries, std::vector<double> &res, int num_threads) const {
    const int n = queries.size();
    res.assign(n * n, 0.0);
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        VID u = queries[i];
        size_t deg_u = get_deg(u);
        for (int j = 0; j < n; j++) {
            VID v = queries[j];
            size_t deg_v = get_deg(v);
            res[i * n + j] = cnt_common_neis_galloping(u, v);
            res[i * n + j] = res[i * n + j] / (double) (deg_u + deg_v - res[i * n + j]);
        }
    }
}

void Graph::solve_jsim_merge(const std::vector<VID> &queries, std::vector<double> &res, int num_threads) const {
    const int n = queries.size();
    res.assign(n * n, 0.0);
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        VID u = queries[i];
        size_t deg_u = get_deg(u);
        for (int j = 0; j < n; j++) {
            VID v = queries[j];
            size_t deg_v = get_deg(v);
            res[i * n + j] = cnt_common_neis_merge(u, v);
            res[i * n + j] = (double) res[i * n + j] / (double) (deg_u + deg_v - res[i * n + j]);
        }
    }
}

void Graph::solve_lcc_galloping(vector<double> &res, const int num_threads) const {
    res.assign(vertex_num, 0.0);
#pragma omp parallel for num_threads(num_threads)
    for (VID u = 0; u < vertex_num; u++) {
        size_t deg_u = get_deg(u);
        if (deg_u < 1) continue;
        for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
            VID v = csr_elist[i];
            res[u] += (double) cnt_common_neis_galloping(u, v);
        }
        res[u] /= (double) (deg_u * (deg_u - 1));
    }
}

void Graph::solve_lcc_merge(vector<double> &res, const int num_threads) const {
    res.assign(vertex_num, 0.0);
#pragma omp parallel for num_threads(num_threads)
    for (VID u = 0; u < vertex_num; u++) {
        size_t deg_u = get_deg(u);
        if (deg_u < 1) continue;
        for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
            VID v = csr_elist[i];
            res[u] += (double) cnt_common_neis_merge(u, v);
        }
        res[u] /= (double) (deg_u * (deg_u - 1));
    }
}

void Graph::solve_sssp(vector<int> &dist, VID source) const {
    dist.clear();
    dist.resize(vertex_num, INT_MAX / 2);
    dist[source] = 0;
    vector<VID> active;
    vector<bool> seen(vertex_num, false);
    active.push_back(source);
    make_heap(active.begin(), active.end(), [&dist](const VID &a, const VID &b) 
        { return dist[a] > dist[b]; });
    while (!active.empty()) {
        pop_heap(active.begin(), active.end(), [&dist](const VID &a, const VID &b) 
            { return dist[a] > dist[b]; });
        VID u = active.back();
        active.pop_back();
        if (seen[u]) continue;
        seen[u] = true;
        for (int i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
            VID v = csr_elist[i];
            if (dist[v] > dist[u] + csr_weight[i]) {
                dist[v] = dist[u] + csr_weight[i];
                active.push_back(v);
                push_heap(active.begin(), active.end(), [&dist](const VID &a, const VID &b) 
                    { return dist[a] > dist[b]; });
            }
        }
    }
}

void Graph::sim_rank(vector<double> &res, const int num_iteration, const int num_threads) {
    res.assign(vertex_num * vertex_num, 0.0);
    for (VID u = 0; u < vertex_num; u++) {
        res[u * vertex_num + u] = 1.0;
    }
    for (int iter = 0; iter < num_iteration; iter++) {
        vector<double> res_new(vertex_num * vertex_num, 0.0);
#pragma omp parallel for num_threads(num_threads)
        for (VID u = 0; u < vertex_num; u++) {
            for (VID v = 0; v < vertex_num; v++) {
                if (u == v) {
                    res[u * vertex_num + v] = 1.0;
                    continue;
                } 
                for (size_t i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
                    VID u_nei = csr_elist[i];
                    for (size_t j = csr_vlist[v]; j < csr_vlist[v+1]; j++) {
                        VID v_nei = csr_elist[j];
                        res_new[u * vertex_num + v] += res[u_nei * vertex_num + v_nei];
                    }
                }
                res_new[u * vertex_num + v] /= (double) get_deg(u) * (double) get_deg(v);
            }
        }
        swap(res, res_new);
    }
}

void Graph::greedy_partition(const size_t partition_number, vector<int> &vertex2partitihon) {
	vector<pair<VID, int>> degree_list(Graph::vertex_num);
	vertex2partitihon.clear();
	for (size_t i = 0; i < Graph::vertex_num; i += 1) {
		vertex2partitihon.push_back(-1);
		degree_list[i] = make_pair((VID)i, Graph::csr_vlist[i + 1] - Graph::csr_vlist[i]);
	}
	sort(degree_list.begin(), degree_list.end(), [](pair<VID, int> a, pair<VID, int> b) {
		return a.second > b.second;
		});
    double _lb = degree_list[0].second / (double)partition_number;
    int _ptr = 0;
    printf("Lower bound: %lf\n", _lb);
	for (pair<VID, int> item : degree_list) {
		vector<int> count_p_neigh(partition_number, 0);
		vector<VID> not_selected;
		VID u = item.first;
		for (size_t i = csr_vlist[u]; i < csr_vlist[u + 1]; i ++) {
			VID cur_v = Graph::csr_elist[i];
			int v_p = vertex2partitihon[(int) cur_v];
			if (v_p > -1) {
				count_p_neigh[v_p] += 1;
			}
			else {
				not_selected.push_back(cur_v);
			}
		}

		if (not_selected.size()) {
			// vector<int> p_list(partition_number, 0);
			// for (size_t i = 0; i < partition_number; i += 1) {
			// 	p_list[i] = max(item.second - count_p_neigh[i] * (int) partition_number, 0);
			// }
			// mt19937 mt(random_device{}());
			// discrete_distribution<> d(p_list.begin(), p_list.end());
			// for (VID v : not_selected) {
			// 	vertex2partitihon[(int)v] = d(mt);
			// }
            shuffle(not_selected.begin(), not_selected.end(), mt19937(random_device{}()));
            vector<int> p_list(partition_number, 0);
            int _u_lb = item.second / partition_number;
            for (size_t i = 0; i < partition_number; i++) {
                p_list[i] = max(0, _u_lb - count_p_neigh[i]);
            }
            int p_list_sum = accumulate(p_list.begin(), p_list.end(), 0);
            size_t ptr = 0;
            if (p_list_sum > 0) {
                for (size_t i = 0; i < partition_number; i++) {
                    size_t len = p_list[i] * not_selected.size() / p_list_sum;
                    for (size_t j = 0; j < len; j++) {
                        if (ptr >= not_selected.size()) cerr << "error at vertex " << u << endl;
                        vertex2partitihon[not_selected[ptr++]] = i;
                    }
                }
            }
            if (ptr < not_selected.size()) {
                for (size_t i = ptr; i < not_selected.size(); i++) {
                    vertex2partitihon[not_selected[i]] = rand() % partition_number;
                }
            }
		}
        vector<int>().swap(count_p_neigh);
        vector<VID>().swap(not_selected);
	}
    for (VID u = 0; u < Graph::vertex_num; u++) {
        if (vertex2partitihon[u] == -1) {
            vertex2partitihon[u] = rand() % partition_number;
        }
    }
}

void Graph::greedy_partition_2(const size_t partition_num, vector<int> &vertex2partition) {
    vector<vector<int>> partition_neis_counts(partition_num, vector<int> (vertex_num, 0));

    vector<pair<VID, int>> degree_list(Graph::vertex_num);
	vertex2partition.clear();
	for (size_t i = 0; i < Graph::vertex_num; i += 1) {
		vertex2partition.push_back(-1);
		degree_list[i] = make_pair((VID)i, Graph::csr_vlist[i + 1] - Graph::csr_vlist[i]);
	}
	sort(degree_list.begin(), degree_list.end(), [](pair<VID, int> a, pair<VID, int> b) {
		return a.second > b.second;
		});
    double _lb = degree_list[0].second / (double) partition_num;
    printf("Lower bound: %lf\n", _lb);

    for (pair<VID, int> item : degree_list) {
        VID u = item.first;
        // printf("-- Vertex: %d, degree: %d.\n", u, item.second);
        int u_part = -1;
        if (csr_vlist[u+1] - csr_vlist[u] > 0) {
            size_t minimax_part_neis = vertex_num;
            for (size_t k = 0; k < partition_num; k++) {
                size_t max_curr_part_neis = 0;
                for (size_t j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
                    VID v = csr_elist[j];
                    if (partition_neis_counts[k][v] > max_curr_part_neis) {
                        max_curr_part_neis = partition_neis_counts[k][v];
                    }
                }
                if (max_curr_part_neis < minimax_part_neis) {
                    minimax_part_neis = max_curr_part_neis;
                    u_part = k;
                }
            }
            // printf("---- u_part: %d, minimax_part_neis: %lu.\n", u_part, minimax_part_neis);
        }
        
        if (u_part == -1) {
            u_part = rand() % partition_num;
        }
        vertex2partition[u] = u_part;
        for (size_t j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
            VID v = csr_elist[j];
            partition_neis_counts[u_part][v]++;
        }
    }
}

Partitioned_Graph::Partitioned_Graph(const Graph &G, const vector<VID> &_vertex2partition, const int _thread_num) {
    vertex2partition = _vertex2partition;
    vertex_num = G.get_vertex_num();
    edge_num = G.get_edge_num();
    partition_num = *max_element(vertex2partition.begin(), vertex2partition.end()) + 1;
    thread_num = _thread_num;

    plist.reserve(edge_num);
    poffsets.resize(partition_num + 1);
  
    vector<vector<VID>> partition_vertices(partition_num);
    for (int i = 0; i < partition_num; i++) partition_vertices[i].reserve(vertex_num);
    for (VID i = 0; i < vertex_num; i++)  partition_vertices[vertex2partition[i]].push_back(i);
    for (int i = 0; i < partition_num; i++) {
        poffsets[i] = plist.size();
        plist.insert(plist.end(), partition_vertices[i].begin(), partition_vertices[i].end());
        vector<VID>().swap(partition_vertices[i]);
    }
    poffsets[partition_num] = vertex_num;
    vector<vector<VID>>().swap(partition_vertices);

    csr_vlist.resize(vertex_num * partition_num + 1);
    csr_elist.reserve(edge_num);
    size_t min_partition_edge_num = edge_num, max_partition_edge_num = 0;
    vector<vector<vector<VID>>> partitioned_adj_list(vertex_num, vector<vector<VID>>(partition_num));
    vector<vector<vector<Weight>>> partitioned_adj_weights(vertex_num, vector<vector<Weight>>(partition_num));
    const EID* _csr_vlist = G.get_csr_vlist();
    const VID* _csr_elist = G.get_csr_elist();
    const Weight* _csr_weight = G.get_csr_weight();
    const bool is_weighted = G.is_weighted();

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        for (EID i = _csr_vlist[u]; i < _csr_vlist[u+1]; i++) {
            VID v = _csr_elist[i];
            partitioned_adj_list[u][vertex2partition[v]].push_back(v);
            if (is_weighted) partitioned_adj_weights[u][vertex2partition[v]].push_back(_csr_weight[i]);
        }
    }
    for (size_t k = 0; k < partition_num; k++) {
        for (VID u = 0; u < vertex_num; u++) {
            csr_vlist[k * vertex_num + u] = csr_elist.size();
            if (partitioned_adj_list[u][k].empty()) continue;
            csr_elist.insert(csr_elist.end(), partitioned_adj_list[u][k].begin(), partitioned_adj_list[u][k].end());
            if (is_weighted) csr_weight.insert(csr_weight.end(), partitioned_adj_weights[u][k].begin(), partitioned_adj_weights[u][k].end());
        }
        size_t partition_edge_num = csr_elist.size() - csr_vlist[k * vertex_num];
        if (partition_edge_num < min_partition_edge_num) min_partition_edge_num = partition_edge_num;
        if (partition_edge_num > max_partition_edge_num) max_partition_edge_num = partition_edge_num;
    }
    csr_vlist[vertex_num * partition_num] = csr_elist.size();
    printf("Partitioned Graph Constructed!\n");
}

size_t Partitioned_Graph::cnt_common_neis_galloping(const VID u, const VID v, const int partition_id) const {
    size_t cnt = 0;
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    size_t v_ptr = csr_vlist[partition_id * vertex_num + v], v_end_ptr = csr_vlist[partition_id * vertex_num + v + 1];
    if (u_end_ptr - u_ptr > v_end_ptr - v_ptr) { 
        swap(u_ptr, v_ptr);
        swap(u_end_ptr, v_end_ptr);
    }
    for (size_t i = u_ptr; i < u_end_ptr; i++) {
        VID target = csr_elist[i];
        size_t offset = lower_bound(csr_elist.begin() + v_ptr, csr_elist.begin() + v_end_ptr, target) - csr_elist.begin();
        if (offset < v_end_ptr && csr_elist[offset] == target) cnt++;
    }
    return cnt;
}

size_t Partitioned_Graph::cnt_common_neis_galloping(const vector<VID> &q, const int partition_id) const {
    size_t cnt = 0, min_deg = vertex_num;
    VID u;
    for (auto v : q) {
        size_t deg_v = csr_vlist[partition_id * vertex_num + v + 1] - csr_vlist[partition_id * vertex_num + v];
        if (deg_v < min_deg) {
            min_deg = deg_v;
            u = v;
        }
    }
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    for (size_t i = u_ptr; i < u_end_ptr; i++) {
        VID target = csr_elist[i];
        bool found = true;
        for (auto v : q) {
            if (v == u) continue;
            size_t v_ptr = csr_vlist[partition_id * vertex_num + v], v_end_ptr = csr_vlist[partition_id * vertex_num + v + 1];
            size_t offset = lower_bound(csr_elist.begin() + v_ptr, csr_elist.begin() + v_end_ptr, target) - csr_elist.begin();
            if (offset >= v_end_ptr || csr_elist[offset] != target) {
                found = false;
                break;
            }
        }
        if (found) cnt++;
    }
    return cnt;
}

size_t Partitioned_Graph::cnt_common_neis_merge(const vector<VID> &q, const int partition_id) const {
    vector<VID> tmp_res1, tmp_res2;
    size_t res_upper_bound = csr_vlist[partition_id * vertex_num + q[0] + 1] - csr_vlist[partition_id * vertex_num + q[0]];
    tmp_res1.reserve(res_upper_bound);
    tmp_res2.reserve(res_upper_bound);
    get_neis(q[0], partition_id, tmp_res1);
    for (size_t i = 1; i < q.size()-1; i ++) {
        get_common_neis(q[i], tmp_res1, partition_id, tmp_res2);
        swap(tmp_res1, tmp_res2);
        tmp_res2.clear();
    }
    size_t res = cnt_common_neis_merge(q[q.size()-1], tmp_res1, partition_id);
    vector<VID>().swap(tmp_res1);
    vector<VID>().swap(tmp_res2);
    return res;
}

size_t Partitioned_Graph::cnt_common_neis_less_than_merge(const VID u, const VID v, const VID threshold, const int partition_id) const {
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    size_t v_ptr = csr_vlist[partition_id * vertex_num + v], v_end_ptr = csr_vlist[partition_id * vertex_num + v + 1];
    size_t cnt = 0;
    VID u_nei = csr_elist[u_ptr], v_nei = csr_elist[v_ptr];
    while (u_ptr < u_end_ptr && v_ptr < v_end_ptr && u_nei < threshold && v_nei < threshold) {
        if (u_nei < v_nei) {
            u_ptr++;
            u_nei = csr_elist[u_ptr];
        } else if (u_nei > v_nei) {
            v_ptr++;
            v_nei = csr_elist[v_ptr];
        } else {
            cnt++;
            u_ptr++;
            v_ptr++;
            u_nei = csr_elist[u_ptr];
            v_nei = csr_elist[v_ptr];
        }
    }
    return cnt;
}

size_t Partitioned_Graph::cnt_common_neis_merge(const VID u, const VID v, const int partition_id) const {
    // size_t u_ptr = csr_vlist[u * partition_num + partition_id], u_end_ptr = csr_vlist[u * partition_num + partition_id + 1];
    // size_t v_ptr = csr_vlist[v * partition_num + partition_id], v_end_ptr = csr_vlist[v * partition_num + partition_id + 1];
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    size_t v_ptr = csr_vlist[partition_id * vertex_num + v], v_end_ptr = csr_vlist[partition_id * vertex_num + v + 1];
    size_t cnt = 0;
    while (u_ptr < u_end_ptr && v_ptr < v_end_ptr) {
        if (csr_elist[u_ptr] < csr_elist[v_ptr]) {
            u_ptr++;
        } else if (csr_elist[u_ptr] > csr_elist[v_ptr]) {
            v_ptr++;
        } else {
            cnt++;
            u_ptr++;
            v_ptr++;
        }
    }
    return cnt;
}

size_t Partitioned_Graph::cnt_common_neis_merge(const VID u, const vector<VID> &nodes, const int partition_id) const {
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    size_t v_ptr = 0, v_end_ptr = nodes.size();
    size_t cnt = 0;
    while (u_ptr < u_end_ptr && v_ptr < v_end_ptr) {
        if (csr_elist[u_ptr] < nodes[v_ptr]) {
            u_ptr++;
        } else if (csr_elist[u_ptr] > nodes[v_ptr]) {
            v_ptr++;
        } else {
            cnt++;
            u_ptr++;
            v_ptr++;
        }
    }
    return cnt;
}

bool value_greater(const pair<VID, VID> &a, const pair<VID, VID> &b) {
    return a.second > b.second;
}

// size_t Partitioned_Graph::cnt_common_neis_merge(const vector<VID> &q, const int partition_id) const {
//     const size_t D = q.size();
//     vector<pair<VID, VID>> head_elements(q.size());
//     for (VID i = 0; i < D; i++) {
//         head_elements[i] = make_pair(i, csr_elist[csr_vlist[partition_id * vertex_num + q[i]]]);
//     }
//     make_heap(head_elements.begin(), head_elements.end(), value_greater);
//     while(head_elements.size() == D) {
//         pop_heap(head_elements.begin(), head_elements.end(), value_greater);
//         VID v = head_elements.back().second;
//     }
// }

size_t Partitioned_Graph::get_edge_num() const {
    return edge_num;
}

size_t Partitioned_Graph::get_partition_num() const {
    return partition_num;
}

size_t Partitioned_Graph::get_vertex_num() const {
    return vertex_num;
}

size_t Partitioned_Graph::solve_friend_triangle_galloping(const VID root, const vector<VID> &nodes) const {
    // vector<size_t> cnts(partition_num, 0);
    size_t res = 0;
    omp_set_num_threads(thread_num);
// #pragma omp parallel for
#pragma omp parallel for reduction(+: res)
    for (size_t i = 0; i < partition_num; i++) {
        for (size_t j = 0; j < nodes.size(); j++) {
            if (nodes[j] == root) continue;
            for (size_t k = j + 1; k < nodes.size(); k++) {
                if (nodes[k] == root) continue;
                size_t local_cnt = cnt_common_neis_galloping({root, nodes[j], nodes[k]}, i);
// #pragma omp atomic
//                 cnts[i] += local_cnt;
                res += local_cnt;
            }
        }
    }
    // return accumulate(cnts.begin(), cnts.end(), 0);
    return res;
}

size_t Partitioned_Graph::solve_friend_triangle_merge(const VID root, const vector<VID> &nodes) const {
    // vector<size_t> cnts(partition_num, 0);
    size_t res = 0;
    omp_set_num_threads(thread_num);
// #pragma omp parallel for
#pragma omp parallel for reduction(+: res)
    for (size_t i = 0; i < partition_num; i++) {
        for (size_t j = 0; j < nodes.size(); j++) {
            if (nodes[j] == root) continue;
            for (size_t k = j + 1; k < nodes.size(); k++) {
                if (nodes[k] == root) continue;
                size_t local_cnt = cnt_common_neis_merge({root, nodes[j], nodes[k]}, i);
// #pragma omp atomic
//                 cnts[i] += local_cnt;
                res += local_cnt;
            }
        }
    }
    // return accumulate(cnts.begin(), cnts.end(), 0);
    return res;
}

// size_t Partitioned_Graph::solve_square_counting() const {
//     size_t res = 0;

// #pragma omp parallel for num_threads(thread_num) reduction(+: res)
//     for (int k = 0; k < partition_num; k++) {
//         for (VID u = 0; u < vertex_num; u++) {
//             for (int i = 0; i < partition_num; i++) {
//                 for (size_t v_ptr = csr_vlist[i * vertex_num + u]; v_ptr < csr_vlist[i * vertex_num + u + 1]; v_ptr++) {
//                     VID v = csr_elist[v_ptr];
//                     if (u <= v) break;
//                     for (int j = 0; j < partition_num; j++) {
//                         for (size_t w_ptr = csr_vlist[j * vertex_num + u]; w_ptr < csr_vlist[j * vertex_num + u + 1]; w_ptr++) {
//                             VID w = csr_elist[w_ptr];
//                             if (v <= w) break;
//                             res += cnt_common_neis_less_than_merge(v, w, w, k);
//                             // res += cnt_common_neis_merge(v, w, k);
//                         }
//                     }
//                 }
//             }        
//         }
//     }
//     return res;
// }

size_t Partitioned_Graph::solve_square_counting() const {
    size_t res = 0;
    vector<pair<VID, VID>> work_items;

    time_counter _t_initial_workitems;
    _t_initial_workitems.start();
    for (VID u = 0; u < vertex_num; u++) {
        for (int i = 0; i < partition_num; i++) {
            for (size_t v_ptr = csr_vlist[i * vertex_num + u]; v_ptr < csr_vlist[i * vertex_num + u + 1]; v_ptr++) {
                VID v = csr_elist[v_ptr];
                if (u <= v) break;
                for (int j = 0; j < partition_num; j++) {
                    for (size_t w_ptr = csr_vlist[j * vertex_num + u]; w_ptr < csr_vlist[j * vertex_num + u + 1]; w_ptr++) {
                        VID w = csr_elist[w_ptr];
                        if (v <= w) break;
                        work_items.push_back(make_pair(v, w));
                    }
                }
            }
        }
    }
    _t_initial_workitems.stop();
    _t_initial_workitems.print("Initial workitems");

#pragma omp parallel for num_threads(thread_num) reduction(+: res)
    for (int k = 0; k < partition_num; k++) {
        for (auto e : work_items) {
            VID v = e.first;
            VID w = e.second;
            res += cnt_common_neis_less_than_merge(v, w, w, k);
        }
    }
    return res;
}

size_t Partitioned_Graph::solve_triangle_counting_galloping() const {
    // vector<size_t> cnts(partition_num, 0);
    size_t res = 0;
    omp_set_num_threads(thread_num);
#pragma omp parallel for reduction(+: res)
    for (size_t i = 0; i < partition_num; i++) {
        for (size_t j = 0; j < partition_num; j++) {
            for (VID u = 0; u < vertex_num; u++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    VID v = csr_elist[k];
                    // size_t local_cnt = cnt_common_neis_galloping(u, v, i);
                    // res += local_cnt;
                    res += cnt_common_neis_galloping(u, v, i);
                }
            }
        }
    }
    return res;
}

size_t Partitioned_Graph::solve_triangle_counting_merge() const {
    size_t res = 0;
#pragma omp parallel for num_threads(thread_num) reduction(+: res)
    for (size_t i = 0; i < partition_num; i++) {
        for (size_t j = 0; j < partition_num; j++) {
            for (VID u = 0; u < vertex_num; u++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    VID v = csr_elist[k];
                    if (v >= u) break;
                    // res += cnt_common_neis_merge(u, v, i);
                    res += cnt_common_neis_less_than_merge(u, v, v, i);
                }
            }
        }
    }
    return res;
}

// pair<int, int> Partitioned_Graph::evaluate_partition() const {
//     int max_part_deg = 0, max_deg = 0;
//     omp_set_num_threads(32);
// #pragma omp parallel for reduction(max : max_part_deg)
//     for (size_t i = 0; i < vertex_num * partition_num; i++) {
//         max_part_deg = max(max_part_deg, csr_vlist[i + 1] - csr_vlist[i]);
//     }
//     vector<int> part_sum(partition_num, 0);
// #pragma omp parallel for
//     for (size_t i = 0; i < partition_num; i++) {
//         for (size_t j = 0; j < vertex_num; j++) {
//             part_sum[i] += csr_vlist[i * vertex_num + j + 1] - csr_vlist[i * vertex_num + j];
//         }
//     }
//     int max_part_sum = *max_element(part_sum.begin(), part_sum.end());
//     return make_pair(max_part_deg, max_part_sum);
// }

void Partitioned_Graph::batch_cnt_common_neis_galloping(const vector<pair<VID, VID>> &queries, vector<size_t>& cnts) const {
    cnts.resize(queries.size(), 0);
    omp_set_num_threads(thread_num);
#pragma omp parallel for
    for (size_t k = 0; k < partition_num; k++) {
        for (size_t i = 0; i < queries.size(); i++) {
            size_t local_cnt = cnt_common_neis_galloping(queries[i].first, queries[i].second, k);
#pragma omp atomic
            cnts[i] += local_cnt;
        }
    }
}

void Partitioned_Graph::batch_cnt_common_neis_galloping(const vector<vector<VID>> &queries, vector<size_t>& cnts) const {
    cnts.resize(queries.size(), 0);
    omp_set_num_threads(thread_num);
#pragma omp parallel for
    for (size_t k = 0; k < partition_num; k++) {
        for (size_t i = 0; i < queries.size(); i++) {
            size_t local_cnt = cnt_common_neis_galloping(queries[i], k);
#pragma omp atomic
            cnts[i] += local_cnt;
        }
    }
}

void Partitioned_Graph::batch_cnt_common_neis_galloping_fine_grained(const vector<pair<VID, VID>> &queries, vector<size_t> &res) const {
    res.resize(queries.size(), 0);
    omp_set_num_threads(thread_num);
#pragma omp parallel for
    for (size_t k = 0; k < partition_num; k++) {
        for (size_t i = 0; i < queries.size(); i++) {
            size_t local_cnt = cnt_common_neis_galloping(queries[i].first, queries[i].second, k);
#pragma omp atomic
            res[i] += local_cnt;
        }
    } 
}

void Partitioned_Graph::batch_cnt_common_neis_merge(const vector<pair<VID, VID>> &queries, vector<size_t>& cnts) const {
    cnts.resize(queries.size(), 0);
    omp_set_num_threads(thread_num);
#pragma omp parallel for
    for (size_t k = 0; k < partition_num; k++) {
        for (size_t i = 0; i < queries.size(); i++) {
            size_t local_cnt = cnt_common_neis_merge(queries[i].first, queries[i].second, k);
#pragma omp atomic
            cnts[i] += local_cnt;
        }
    }
}

void Partitioned_Graph::batch_cnt_common_neis_merge(const vector<vector<VID>> &queries, vector<size_t>& cnts) const {
    cnts.resize(queries.size(), 0);
    omp_set_num_threads(thread_num);
#pragma omp parallel for
    for (size_t k = 0; k < partition_num; k++) {
        for (size_t i = 0; i < queries.size(); i++) {
            size_t local_cnt = cnt_common_neis_merge(queries[i], k);
#pragma omp atomic
            cnts[i] += local_cnt;
        }
    }
}

void Partitioned_Graph::batch_cnt_common_neis_merge_fine_grained(const vector<pair<VID, VID>> &queries, vector<size_t> &res) const {
    res.resize(queries.size(), 0);
    omp_set_num_threads(thread_num);
#pragma omp parallel for
    for (size_t k = 0; k < partition_num; k++) {
        for (size_t i = 0; i < queries.size(); i++) {
            size_t local_cnt = cnt_common_neis_merge(queries[i].first, queries[i].second, k);
#pragma omp atomic
            res[i] += local_cnt;
        }
    } 
}

void Partitioned_Graph::evaluate_partition() const {
    vector<int> degs(vertex_num, 0);
    int max_part_deg = 0, max_deg = 0; 
    for (VID u = 0; u < vertex_num; u++) {
        for (int i = 0; i < partition_num; i++) {
            int part_deg = csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
            degs[u] += part_deg;
            max_part_deg = max(max_part_deg, part_deg);
        }
        max_deg = max(max_deg, degs[u]);
    }
    double balance_ratio = (double) (partition_num * max_part_deg) / (double) max_deg;
    printf("max part deg: %d, max deg: %d, balance ratio: %f\n", max_part_deg, max_deg, balance_ratio);
}

void Partitioned_Graph::get_common_neis(const VID u, const vector<VID> &nodes, const int partition_id, vector<VID>& res) const {
    res.clear();
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    size_t v_ptr = 0, v_end_ptr = nodes.size();
    res.reserve(std::min(u_end_ptr - u_ptr, v_end_ptr - v_ptr));
    while (u_ptr < u_end_ptr && v_ptr < v_end_ptr) {
        if (csr_elist[u_ptr] == nodes[v_ptr]) {
            res.push_back(csr_elist[u_ptr]);
            u_ptr++;
            v_ptr++;
        }
        else if (csr_elist[u_ptr] < nodes[v_ptr]) {
            u_ptr++;
        }
        else {
            v_ptr++;
        }
    }
}

void Partitioned_Graph::get_neis(const VID u, const int partition_id, vector<VID> &res) const {
    size_t start_ptr = csr_vlist[partition_id * vertex_num + u], end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    res.assign(csr_elist.begin() + start_ptr, csr_elist.begin() + end_ptr);
}

// void Partitioned_Graph::page_rank(vector<float> &pr_vals, float damping_factor, int num_iterations) const {
//     vector<float> pr_vals_new(vertex_num);
//     vector<float> degs(vertex_num, 0.0);
//     float _damping = (1 - damping_factor) / vertex_num;
//     printf("Basic damping score: %f\n", _damping);
//     vector<time_counter> local_timers(partition_num);
//     time_counter _t_pull, _t_reset;
// #pragma omp declare reduction(vec_add : vector<float> : \
//     std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>()) ) \
//     initializer(omp_priv = vector<float>(omp_orig.size(), 0.0))

// #pragma omp parallel for num_threads(thread_num)
//     for (VID u = 0; u < vertex_num; u++) {
//         for (int i = 0; i < partition_num; i++) {
//             degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
//         }
//         degs[u] /= damping_factor;
//         pr_vals[u] = degs[u] / vertex_num;
//     }
    
//     for (int iter = 0; iter < num_iterations; iter++) {
//         _t_reset.start();
// #pragma omp parallel for num_threads(thread_num)
//         for (VID u = 0; u < vertex_num; u++) {
//             pr_vals_new[u] = _damping;
//             pr_vals[u] /= degs[u]; 
//         }
//         _t_reset.stop();

//         _t_pull.start();
// #pragma omp parallel for num_threads(thread_num) reduction(vec_add:pr_vals_new)
//         for (int i = 0; i < partition_num; i++)
//         {
//             local_timers[i].start();
//             float local_pr_update = 0.0;
//             for (VID u = 0; u < vertex_num; u++) {
//                 for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
//                     VID v = csr_elist[j];
//                     // pr_vals_new[u] += pr_vals[v];
//                     local_pr_update += pr_vals[v];
//                 }
//                 pr_vals_new[u] += local_pr_update;
//             }
//             local_timers[i].stop();
//         }
//         _t_pull.stop();
//         swap(pr_vals, pr_vals_new);
//     }
//     _t_reset.print("pagerank reset");
//     _t_pull.print("pagerank pull");
//     for (int i = 0; i < partition_num; i++) {
//         local_timers[i].print("local pagerank pull");
//     }
// }

void Partitioned_Graph::page_rank(vector<float> &pr_vals, float damping_factor, int num_iterations) const {
    vector<float> pr_vals_new(vertex_num, 0.0);
    vector<float> degs(vertex_num, 0.0);
    float _damping = (1 - damping_factor) / vertex_num;
    printf("Basic damping score: %f\n", _damping);
    // vector<time_counter> local_timers(thread_num);
    time_counter _t_push, _t_reset;

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        for (int i = 0; i < partition_num; i++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
        degs[u] /= damping_factor;
        pr_vals[u] = degs[u] / vertex_num;
    }
    
    for (int iter = 0; iter < num_iterations; iter++) {
        _t_reset.start();
#pragma omp parallel for num_threads(thread_num)
        for (VID u = 0; u < vertex_num; u++) pr_vals[u] /= degs[u]; 
        fill(pr_vals_new.begin(), pr_vals_new.end(), _damping);
        _t_reset.stop();

        _t_push.start();
#pragma omp parallel for num_threads(thread_num) schedule(dynamic)
        for (int i = 0; i < partition_num; i++)
        {
            int tid = omp_get_thread_num();
            // local_timers[tid].start();
            for (VID u = 0; u < vertex_num; u++) {
            // for (VID u : non_empty_projections[i]) {
                for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
                    VID v = csr_elist[j];
                    pr_vals_new[v] += pr_vals[u];
                }
            }
            // local_timers[tid].stop();
        }
        _t_push.stop();
        swap(pr_vals, pr_vals_new);
    }
    _t_reset.print("pagerank reset");
    _t_push.print("pagerank pull");
    // for (int i = 0; i < thread_num; i++) {
    //     local_timers[i].print("local pagerank pull");
    // }
}

void Partitioned_Graph::page_rank_residual(vector<float> &pr_vals, float damping_factor, int num_iterations) const {
    const float epsilon = 1e-6;
    vector<float> pr_vals_new(vertex_num, 0.0);
    vector<float> degs(vertex_num, 0.0);
    vector<bool> active(vertex_num, true);
    float _damping = (1 - damping_factor) / vertex_num;
    printf("Basic damping score: %f\n", _damping);
    vector<time_counter> local_timers(thread_num);
    time_counter _t_pull, _t_reset;

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        for (int i = 0; i < partition_num; i++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
        pr_vals[u] = 1.0 / vertex_num;
        pr_vals_new[u] = _damping;
    }

    vector<VID> Frontier, Frontier_new;
    Frontier.reserve(vertex_num);
    Frontier_new.reserve(vertex_num);
    for (VID u = 0; u < vertex_num; u++) Frontier.push_back(u);
    
    for (int iter = 0; iter < num_iterations; iter++) {
        _t_pull.start();
#pragma omp parallel for num_threads(thread_num) schedule(dynamic)
        for (int i = 0; i < partition_num; i++)
        {
            int tid = omp_get_thread_num();
            local_timers[tid].start();
            for (VID u = 0; u < vertex_num; u++) {
                float local_update = damping_factor * pr_vals[u] / degs[u];
                for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
                    VID v = csr_elist[j];
                    if (active[v]) pr_vals_new[v] += local_update;
                }
            }
            local_timers[tid].stop();
        }
        _t_pull.stop();
        swap(pr_vals, pr_vals_new);

        _t_reset.start();
        for (VID u : Frontier) 
        {
            if (abs(pr_vals[u] - pr_vals_new[u]) < epsilon) active[u] = false;
            else Frontier_new.push_back(u);
        }
        swap(Frontier, Frontier_new);
        Frontier_new.clear();
        fill(pr_vals_new.begin(), pr_vals_new.end(), _damping);
        _t_reset.stop();
    }
    _t_reset.print("pagerank reset");
    _t_pull.print("pagerank pull");
    for (int i = 0; i < thread_num; i++) {
        local_timers[i].print("local pagerank pull");
    }
}

void Partitioned_Graph::reorder() {
    vector<VID> new2origin(vertex_num);
    vector<VID> origin2new(vertex_num);
    // Obtain the new vertex ordering.
    VID curr_new_id = 0;
    for (int i = 0; i < partition_num; i++) {
        for (int j = poffsets[i]; j < poffsets[i + 1]; j++) {
            VID u = plist[j];
            new2origin[curr_new_id] = u;
            origin2new[u] = curr_new_id++;
        }
    }

    // Reorder the CSR according to the obtained ordering.
    vector<EID> csr_vlist_new(partition_num * vertex_num + 1);
    vector<VID> csr_elist_new;
    csr_elist_new.reserve(csr_elist.size());
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            csr_vlist_new[i * vertex_num + u] = csr_elist_new.size();
            VID u_origin_id = new2origin[u];
            size_t _l = csr_elist_new.size();
            for (size_t j = csr_vlist[i * vertex_num + u_origin_id]; j < csr_vlist[i * vertex_num + u_origin_id + 1]; j++) {
                csr_elist_new.push_back(origin2new[csr_elist[j]]);
            }
            // sort(csr_elist_new.begin() + _l, csr_elist_new.end());
        }
    }
    csr_vlist_new[partition_num * vertex_num] = csr_elist_new.size();
    csr_vlist.swap(csr_vlist_new);
    csr_elist.swap(csr_elist_new);
    // Free space.
    vector<EID>().swap(csr_vlist_new);
    vector<VID>().swap(csr_elist_new);
}

void Partitioned_Graph::solve_bfs(VID u, vector<VID> &rank) const {
    rank.clear();
    rank.resize(vertex_num, INT_MAX);
    rank[u] = -1;

    int round = 0;
    vector<VID> frontiers;
    frontiers.reserve(vertex_num);
    frontiers.push_back(u);
    boost::dynamic_bitset<> next_frontiers(vertex_num);

    bool __pull = false;
    int seen_cnt = 0;

    while (!frontiers.empty()) {
        round++;
        if (__pull) {
#pragma omp parallel for num_threads(thread_num)
            for (int i = 0; i < partition_num; i++) {
                for (VID curr : frontiers) {
                    if (rank[curr] != INT_MAX) continue;
                    for (size_t ptr = csr_vlist[curr]; ptr < csr_vlist[curr + 1]; ptr++) {
                        VID v = csr_elist[ptr];
                        if (rank[v] != INT_MAX) {
                            rank[curr] = round;
                            next_frontiers[curr] = false;
                            break;
                        }
                    }
                }
            }  
        }
        else {
#pragma omp parallel for num_threads(thread_num)
            for (int i = 0; i < partition_num; i++) {
                for (VID curr : frontiers) {
                    for (size_t ptr = csr_vlist[i * vertex_num + curr]; ptr < csr_vlist[i * vertex_num + curr + 1]; ptr++) { 
                        VID v = csr_elist[ptr];
                        if (rank[v] == INT_MAX) {
                            rank[v] = round;
                            next_frontiers.set(v);
                        }
                    }     
                }
            }
            seen_cnt += frontiers.size();
            if (seen_cnt > vertex_num / 5) {
                __pull = true;
                next_frontiers.reset();
                for (VID u = 0; u < vertex_num; u++) {
                    if (rank[u] == INT_MAX) 
                        next_frontiers.set(u);
                }
            }
        }      
        frontiers.clear();
        for (VID curr = next_frontiers.find_first(); curr != boost::dynamic_bitset<>::npos; curr = next_frontiers.find_next(curr)) {
            frontiers.push_back(curr);
        }
        if (__pull == false) next_frontiers.reset();
    }
}

// void Partitioned_Graph::solve_bfs(VID u, vector<VID> &bfs_rank) const {
//     bfs_rank.clear();
//     bfs_rank.resize(vertex_num, INT_MAX);
//     bfs_rank[u] = 0;

//     int round = 1;
//     boost::dynamic_bitset<> frontiers(vertex_num);
//     boost::dynamic_bitset<> next_frontiers(vertex_num);
//     frontiers.set(u);

//     vector<VID> __frontiers;
//     __frontiers.reserve(vertex_num);
//     __frontiers.push_back(u);
//     time_counter _t;

//     while (frontiers.any()) {
// #pragma omp parallel for num_threads(thread_num)
//         for (int i = 0; i < partition_num; i++) {
//             for (VID curr : __frontiers) {
//                 for (size_t ptr = csr_vlist[i * vertex_num + curr]; ptr < csr_vlist[i * vertex_num + curr + 1]; ptr++) {
//                     VID v = csr_elist[ptr];
//                     if (bfs_rank[v] == INT_MAX) {
//                         bfs_rank[v] = round;
//                         next_frontiers.set(v);
//                     }
//                 }
//             }
//         }
//         swap(frontiers, next_frontiers);
//         next_frontiers.reset();
//         round++;
//         _t.start();
//         __frontiers.clear();
//         for (VID curr = frontiers.find_first(); curr != boost::dynamic_bitset<>::npos; curr = frontiers.find_next(curr)) {
//             __frontiers.push_back(curr);
//         }
//         _t.stop();
//     }
//     _t.print("enumerate frontiers");
// }

// void Partitioned_Graph::solve_bfs(VID u, vector<VID> &parents) const {
//     parents.clear();
//     parents.resize(vertex_num, INT_MAX);
//     parents[u] = -1;

//     vector<vector<VID>> curr_frontiers(partition_num, vector<VID>());
//     vector<vector<VID>> next_frontiers(partition_num, vector<VID>());
// #pragma omp parallel for num_threads(thread_num)
//     for (int i = 0; i < partition_num; i++) {
//         curr_frontiers[i].reserve(partition_cnts[i]);
//         next_frontiers[i].reserve(partition_cnts[i]);
//         // frontiers[i] = vector<VID>(partition_cnts[i], -1);
//     }

//     curr_frontiers[vertex2partition[u]].push_back(u);
//     bool is_finished = false;
//     while (is_finished == false) {
//         is_finished = true;
// #pragma omp parallel for num_threads(thread_num)
//         for (int i = 0; i < partition_num; i++) {
//             next_frontiers[i].clear();
//             for (int j = 0; j < partition_num; j++) {
//                 for (auto curr : curr_frontiers[j]) {
//                     for (size_t ptr = csr_vlist[i * vertex_num + curr]; ptr < csr_vlist[i * vertex_num + curr + 1]; ptr++) {
//                         VID v = csr_elist[ptr];
//                         if (parents[v] == INT_MAX) {
//                             parents[v] = curr;
//                             next_frontiers[i].push_back(v);
//                         }
//                     }
//                 }
//             }
//             if (!next_frontiers[i].empty()) is_finished = false;
//         }
//         swap(curr_frontiers, next_frontiers);
//     }
// }

void Partitioned_Graph::solve_cdlp(vector<int> &labels, int max_iteration) const {
#pragma omp declare reduction(vec_max : vector<int> : \
        std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), [](int a, int b) {return max(a, b);}) ) \
        initializer(omp_priv = vector<int>(omp_orig.size(), 0))
    
    for (int iter = 0; iter < max_iteration; iter++) {
        vector<int> tmp_labels(vertex_num, 0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_max : tmp_labels)
        for (int i = 0; i < partition_num; i++) {
            for (VID u = 0; u < vertex_num; u++) {
                if (csr_vlist[i * vertex_num + u + 1] == csr_vlist[i * vertex_num + u]) continue;
                for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++)
                    tmp_labels[u] = max(tmp_labels[u], labels[csr_elist[j]]);
            }
        }
        // if (tmp_labels == labels) break;
        labels.swap (tmp_labels);
    }
}

void Partitioned_Graph::solve_kcore(vector<int> &core_numbers) const {
    vector<vector<VID>> frontiers;
    vector<vector<VID>> to_removes;
    vector<vector<VID>> remainings;
    frontiers.resize(partition_num, vector<VID>());
    to_removes.resize(partition_num, vector<VID>());
    remainings.resize(partition_num, vector<VID>());
    for (int i = 0; i < partition_num; i++) {
        int pcnt = poffsets[i+1] - poffsets[i];
        frontiers[i].reserve(pcnt);
        to_removes[i].reserve(pcnt);
        remainings[i].reserve(pcnt);
    }
    for (VID u = 0; u < vertex_num; u++) {
        frontiers[vertex2partition[u]].push_back(u);
    }

    vector<int> degrees(vertex_num, 0);
    core_numbers.resize(vertex_num, 0);
//     const int *degs = degrees.data();
#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        for (int i = 0; i < partition_num; i++) {
            degrees[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

    size_t largest_core = 0;
    bool is_inner_loop_finished = false;
    bool is_outer_loop_finished = false;
    for (size_t k = 1; k <vertex_num; k++) {
        is_outer_loop_finished = true;
        is_inner_loop_finished = false;
        while (!is_inner_loop_finished) {
            is_inner_loop_finished = true;
            int min_remaining_degree = vertex_num;
#pragma omp parallel for num_threads(thread_num) reduction(min:min_remaining_degree)
            for (int i = 0; i < partition_num; i++) {
                to_removes[i].clear();
                remainings[i].clear();
                for (VID u : frontiers[i]) {
                    if (degrees[u] < k) {
                        core_numbers[u] = k-1;
                        degrees[u] = 0;
                        to_removes[i].emplace_back(u);
                    }
                    else {
                        remainings[i].emplace_back(u);
                        min_remaining_degree = min(min_remaining_degree, degrees[u]);
                    }
                }
                if (!remainings[i].empty()) is_outer_loop_finished = false;
                if (!to_removes[i].empty()) is_inner_loop_finished = false;             
            }

            swap(frontiers,remainings);  

            if (is_inner_loop_finished) {
                k = min_remaining_degree;
                break;
            } 
#pragma omp parallel for num_threads(thread_num)
            for (int i = 0; i < partition_num; i++) {
                for (int k = 0; k < partition_num; k++) {
                    for (VID u : to_removes[k]) {
                        for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
                            VID v = csr_elist[j];
                            degrees[v]--;
                        }
                    }
                }
            }
        }
        if (is_outer_loop_finished) {
            largest_core = k-1;
            break;
        }
    }
    printf("largest core: %lu.\n", largest_core);
}

void Partitioned_Graph::solve_mis(vector<int> &res) const {
    enum {CONDITIONALLY_IN, UNDICEDED, OUT, IN};
    vector<int> flags(vertex_num, CONDITIONALLY_IN);
    size_t round = 0;
    boost::dynamic_bitset<> frontier(vertex_num);
    boost::dynamic_bitset<> next_frontier(vertex_num);
    auto npos = boost::dynamic_bitset<>::npos;
    size_t num_blocks = frontier.num_blocks();
    size_t block_per_thread = num_blocks / thread_num;
    time_counter _t_pull, _t_filter, _t_reset;

    _t_pull.start();

#pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
                VID v = csr_elist[j];
                if (flags[v] == IN) {if (flags[u] != OUT) flags[u] = OUT;}
                else if (v < u && flags[u] == CONDITIONALLY_IN && flags[v] < OUT) {
                    flags[u] = UNDICEDED;
                }
            } 
        }
    }

    _t_pull.stop();

    _t_filter.start();

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        if (flags[u] == CONDITIONALLY_IN) {flags[u] = IN;}
        else if (flags[u] == UNDICEDED) {
            flags[u] = CONDITIONALLY_IN;
            frontier.set(u);
        } 
    }

    _t_filter.stop();

    while (frontier.any()) {
        round++;

        _t_pull.start();

#pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < partition_num; i++) {
            auto curr = frontier.find_first();
            while(curr != npos) {
                // if (flags[curr] == OUT) { curr = frontier.find_next(curr); continue; }
                // ##### Pull-based algorithm #####
                for (size_t j = csr_vlist[i * vertex_num + curr]; j < csr_vlist[i * vertex_num + curr + 1]; j++) {
                    VID v = csr_elist[j];
                    if (flags[v] == IN) {if (flags[curr] != OUT) flags[curr] = OUT; break;}
                    else if (v < curr && flags[curr] == CONDITIONALLY_IN && flags[v] < OUT) {
                        flags[curr] = UNDICEDED;
                    }
                }
                curr = frontier.find_next(curr);
            }
        }

        _t_pull.stop();

        _t_filter.start();

        int is_unfinished = 0;
#pragma omp parallel for num_threads(thread_num) reduction(+: is_unfinished)
        for (int i = 0; i < partition_num; i++) {
            size_t _left_ptr = i * vertex_num / partition_num;
            size_t _right_ptr = min((i + 1) * vertex_num / partition_num, vertex_num);
            auto curr = (frontier[_left_ptr] == 1) ? _left_ptr : frontier.find_next(_left_ptr);
            while (curr != npos && curr < _right_ptr) {
                if (flags[curr] == CONDITIONALLY_IN) {flags[curr] = IN; is_unfinished = 1;}
                else if (flags[curr] == UNDICEDED) {
                    flags[curr] = CONDITIONALLY_IN;
                    next_frontier.set(curr);
                } 
                curr = frontier.find_next(curr);
            }
        }

        _t_filter.stop();  

        if (is_unfinished == 0) break;
        swap(frontier, next_frontier);
        _t_reset.start();
        next_frontier.reset();
        _t_reset.stop();
    }

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        res[u] = (flags[u] == IN);
    }
    printf("round: %lu\n", round);
    _t_pull.print("pull");
    _t_filter.print("filter");
    _t_reset.print("reset frontiers");
}


// void Partitioned_Graph::solve_mis(vector<int> &res) const {
//     enum {UNDICEDED, CONDITIONALLY_IN, OUT, IN};
//     vector<int> flags(vertex_num, CONDITIONALLY_IN);
//     size_t round = 0;
//     // size_t res_cnt = 0;
//     vector<vector<VID>> frontiers(partition_num);
//     vector<vector<VID>> next_frontiers(partition_num);
// #pragma omp parallel for num_threads(thread_num)
//     for (int i = 0; i < partition_num; i++) {
//         frontiers[i].reserve(poffsets[i+1] - poffsets[i]);
//         next_frontiers[i].reserve(poffsets[i+1] - poffsets[i]);
//     }

// #pragma omp parallel for num_threads(thread_num)
//     for (int i = 0; i < partition_num; i++) {
//         for (VID u = 0; u < vertex_num; u++) {
//             for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
//                 VID v = csr_elist[j];
//                 if (flags[v] == IN) {if (flags[u] != OUT) flags[u] = OUT;}
//                 else if (v < u && flags[u] == CONDITIONALLY_IN && flags[v] < OUT) {
//                     flags[u] = UNDICEDED;
//                 }
//             } 
//         }
//     }
//     bool is_finished = true;
// #pragma omp parallel for num_threads(thread_num)
//     for (int i = 0; i < partition_num; i++) {
//         for (VID u = 0; u < vertex_num; u++) {
//             if (vertex2partition[u] != i) continue; 
//             if (flags[u] == CONDITIONALLY_IN) {flags[u] = IN;}
//             else if (flags[u] == UNDICEDED) {
//                 flags[u] = CONDITIONALLY_IN;
//                 frontiers[i].push_back(u);
//                 is_finished = false;
//             } 
//         }
//     }

//     while (!is_finished) {
//         round++;
//         is_finished = true;
// #pragma omp parallel for num_threads(thread_num)
//         for (int i = 0; i < partition_num; i++) {
//             for (int k = 0; k < partition_num; k++) {
//                 for (auto u : frontiers[k]) {
//                     for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
//                         VID v = csr_elist[j];
//                         if (flags[v] == IN) {if (flags[u] != OUT) flags[u] = OUT; break;}
//                         else if (v < u && flags[u] == CONDITIONALLY_IN && flags[v] < OUT) {
//                             flags[u] = UNDICEDED;
//                         }
//                     }
//                 }
//             }
//         }
// #pragma omp parallel for num_threads(thread_num)
//         for (int i = 0; i < partition_num; i++) {
//             next_frontiers[i].clear();
//             for (auto u : frontiers[i]) {
//                 if (flags[u] == CONDITIONALLY_IN) {flags[u] = IN;}
//                 else if (flags[u] == UNDICEDED) {
//                     flags[u] = CONDITIONALLY_IN;
//                     next_frontiers[i].push_back(u);
//                     is_finished = false;
//                 } 
//             }
//         }
        
//         // printf("round: %lu, independent set size: %lu.\n", round, res_cnt);
//         swap(frontiers, next_frontiers);
//     }

// #pragma omp parallel for num_threads(thread_num)
//     for (VID u = 0; u < vertex_num; u++) {
//         res[u] = (flags[u] == IN);
//     }
//     printf("round: %lu\n", round);
// }

void Partitioned_Graph::solve_jsim_galloping(const std::vector<VID> &queries, std::vector<double> &res) const {
    const int n = queries.size();
    res.assign(n * n, 0.0);
#pragma omp declare reduction(vec_add : vector<double> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) ) \
    initializer(omp_priv = vector<double>(omp_orig.size(), 0.0))

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num) reduction(vec_add:res)
    for (int k = 0; k < partition_num; k++) {
        for (int i = 0; i < n; i++) {
            VID u = queries[i];
            for (int j = 0; j < n; j++) {
                VID v = queries[j];
                size_t common_neis = cnt_common_neis_galloping(u, v, k);
                res[i * n + j] += (double) common_neis;
            }
        }
    }
    
#pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = (degs[i] + degs[j] - res[i * n + j]);
        }
    }
}

void Partitioned_Graph::solve_jsim_merge(const std::vector<VID> &queries, std::vector<double> &res) const {
    const int n = queries.size();
    res.assign(n * n, 0.0);
#pragma omp declare reduction(vec_add : vector<double> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) ) \
    initializer(omp_priv = vector<double>(omp_orig.size(), 0.0))

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num) reduction(vec_add:res)
    for (int k = 0; k < partition_num; k++) {
        for (int i = 0; i < n; i++) {
            VID u = queries[i];
            for (int j = 0; j < n; j++) {
                VID v = queries[j];
                size_t common_neis = cnt_common_neis_merge(u, v, k);
                res[i * n + j] += (double) common_neis;
            }
        }
    }
    
#pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = (degs[i] + degs[j] - res[i * n + j]);
        }
    }
}

void Partitioned_Graph::solve_lcc_galloping(vector<double> &res) const {
#pragma omp declare reduction(vec_add : vector<double> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) ) \
    initializer(omp_priv = vector<double>(omp_orig.size(), 0.0))

    res.assign(vertex_num, 0.0);
    #pragma omp parallel for num_threads(thread_num) reduction(vec_add:res)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            for (int j = 0; j < partition_num; j++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    VID v = csr_elist[k];
                    res[u] += cnt_common_neis_galloping(u, v, i);
                }
            }
        }
    }

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        if (degs[u] < 0.5) continue;
        res[u] /= (degs[u] * (degs[u] - 1));
    }
}

void Partitioned_Graph::solve_lcc_merge(vector<double> &res) const {
#pragma omp declare reduction(vec_add : vector<double> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) ) \
    initializer(omp_priv = vector<double>(omp_orig.size(), 0.0))

    res.assign(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:res)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            for (int j = 0; j < partition_num; j++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    VID v = csr_elist[k];
                    res[u] += cnt_common_neis_merge(u, v, i);
                }
            }
        }
    }

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (VID u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num)
    for (VID u = 0; u < vertex_num; u++) {
        if (degs[u] < 0.5) continue;
        res[u] /= (degs[u] * (degs[u] - 1));
    }
} 

void Partitioned_Graph::solve_sssp(VID src, vector<int> &dist)  const {
    dist[src] = 0;

    vector<size_t> workloads(partition_num, 0);

    bool is_finished = false;
    boost::dynamic_bitset<> updated(vertex_num, false);
    vector<VID> frontiers;
    frontiers.reserve(vertex_num);
    frontiers.push_back(src);
    int round = 0;

    while (!is_finished)
    {
        is_finished = true;
        // printf("Round = %d, Forntier.size = %lu.\n", round++, frontiers.size());
#pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < partition_num; i++) {
            const int tid = omp_get_thread_num();
            for (VID u : frontiers) {
                workloads[i] +=  (csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u]);
                for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
                    VID v = csr_elist[j];
                    if (dist[u] + csr_weight[j] < dist[v]) {
                        dist[v] = dist[u] + csr_weight[j];
                        updated.set(v);
                        is_finished = false;
                    }
                }
            }
        }
        // printf("Updated %lu vertices!\n", updated.count());
        frontiers.clear();
        for (auto curr = updated.find_first(); curr != boost::dynamic_bitset<>::npos; curr = updated.find_next(curr)) {
            frontiers.push_back(curr);
        }
        updated.reset();
    }

    size_t max_workload = 0, min_workload = INT_MAX;
    for (int i = 0; i < partition_num; i++) {
        max_workload = max(max_workload, workloads[i]);
        min_workload = min(min_workload, workloads[i]);
    }
    printf("Max workload: %lu, Min workload: %lu\n", max_workload, min_workload);
    printf("Workload imblance ratio: %.3f\n", (float) max_workload / (float) min_workload);
}