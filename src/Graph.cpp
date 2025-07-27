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


// Operators of Graph class
Graph::Graph(const string graphfile) {
    bool has_zero = false;
	vertex_num = 0;
	edge_num = 0;
	FILE *fp = fopen(graphfile.c_str(), "r");
	if (fp == NULL) {
        std::cout << "fail to open " << graphfile << std::endl;
    }
	vector<pair<node, node>> edge_list;

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
        edge_list.push_back({(node) u, (node) v});
    }    
    fclose(fp);

	if (has_zero) vertex_num++;
    printf("Vertex num: %lu.\n", vertex_num);
    vector<vector<node>> adj_list(vertex_num);  
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
    vector<pair<node, node>>().swap(edge_list);
	for (int i = 0; i < vertex_num; ++i){
		if (!adj_list[i].empty()) {
			sort(adj_list[i].begin(), adj_list[i].end());
			vector<node> tmp_neis(adj_list[i]);
			adj_list[i].clear();
			node prev = INT_MAX;
			for (auto j : tmp_neis) {
				if (j == prev) continue;
				prev = j;
				adj_list[i].push_back(j);
			}
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
    vector<vector<node>>().swap(adj_list);

	printf("Number of nodes: %lu, number of edges: %lu, size of adj-list: %.3f MB!\n", 
			vertex_num, edge_num, (float)(edge_num + vertex_num) * sizeof(node) / (1<<20));
}

Graph::Graph(const string csr_vlist_binfile, const string csr_elist_binfile) {
    ifstream input_csr_vlist(csr_vlist_binfile, ios::binary);
    if (!input_csr_vlist.is_open()) {
        cerr << "Unable to open csr vlist binfile from " << csr_vlist_binfile << endl;
        return;
    }
    input_csr_vlist.seekg(0, ios::end);
    streamsize size = input_csr_vlist.tellg();
    input_csr_vlist.seekg(0, ios::beg);
    csr_vlist.resize(size / sizeof(int));
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
    csr_elist.resize(size / sizeof(node));
    input_csr_elist.read(reinterpret_cast<char*>(csr_elist.data()), size);
    input_csr_elist.close();
    edge_num = csr_elist.size();
    printf("Has read graph! #Vertices: %lu, #Edges: %lu.\n", vertex_num, edge_num);
}

Graph::Graph(const vector<vector<node>> &adj_list) {
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

const int* Graph::get_csr_vlist() const {
    return csr_vlist.data();
}

const node* Graph::get_csr_elist() const {
    return csr_elist.data();
}

bool Graph::has_edge(node u, node v) const {
    return get_deg(u) < get_deg(v) ? has_directed_edge(u, v) : has_directed_edge(v, u);
}

bool Graph::has_directed_edge(node u, node v) const {
    size_t ptr = lower_bound(csr_elist.begin() + csr_vlist[u], csr_elist.begin() + csr_vlist[u+1], v) - csr_elist.begin();
    return ptr < csr_vlist[u+1] && csr_elist[ptr] == v;
}

node Graph::get_nei(node u, size_t offset) const {
    return csr_elist[csr_vlist[u] + offset];
}

pair<node,node> Graph::get_edge(size_t offset) const {
    node v = csr_elist[offset];
    node u = upper_bound(csr_vlist.begin(), csr_vlist.end(), offset) - csr_vlist.begin() - 1;
    return {u, v};
}

size_t Graph::cnt_common_neis_galloping(node u, node v) const {
    size_t u_start_ptr = csr_vlist[u], u_end_ptr = csr_vlist[u+1], v_start_ptr = csr_vlist[v], v_end_ptr = csr_vlist[v+1], res = 0;
    if (u_end_ptr - u_start_ptr > v_end_ptr - v_start_ptr) {
        swap(u_start_ptr, v_start_ptr);
        swap(u_end_ptr, v_end_ptr);
    }
    for (size_t i = u_start_ptr; i < u_end_ptr; i ++) {
        node target = csr_elist[i];
        size_t offset = lower_bound(csr_elist.begin() + v_start_ptr, csr_elist.begin() + v_end_ptr, target) - csr_elist.begin();
        if (offset < v_end_ptr && csr_elist[offset] == target) res ++;
    }
    return res;
}

size_t Graph::cnt_common_neis_galloping(const vector<node> &q) const {
    size_t res = 0, min_deg = vertex_num;
    node u;
    for (node v: q) {
        if (get_deg(v) < min_deg) {
            min_deg = get_deg(v);
            u = v;
        }
    }
    size_t u_start_ptr = csr_vlist[u], u_end_ptr = csr_vlist[u+1];
    for (size_t i = u_start_ptr; i < u_end_ptr; i ++) {
        node target = csr_elist[i];
        bool found = true;
        for (node v: q) {
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

size_t Graph::cnt_common_neis_merge(node u, node v) const {
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

size_t Graph::cnt_common_neis_merge(node u, const vector<node> &nodes) const {
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

size_t Graph::get_edge_num() const {
    return edge_num;
}

size_t Graph::get_vertex_num() const {
    return vertex_num;
}

size_t Graph::get_deg(node u) const {
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

ull Graph::size_in_bytes() const {
    return csr_vlist.size() * sizeof(int) + csr_elist.size() * sizeof(node);
}

void Graph::get_common_neis(node u, node v, vector<node>& res) const {
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

void Graph::get_common_neis(node u, const vector<node> &nodes, vector<node> &res) const {
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
    for (node u = 0; u < vertex_num; u++) {
        degs[u] = csr_vlist[u+1] - csr_vlist[u];
    }
}

void Graph::get_neis(node u, vector<node> &neis) const {
    neis.clear();
    if (csr_vlist[u] == csr_vlist[u+1]) return;
    neis.reserve(csr_vlist[u+1] - csr_vlist[u]);
    neis.insert(neis.end(), csr_elist.begin() + csr_vlist[u], csr_elist.begin() + csr_vlist[u+1]);
}

void Graph::get_two_hop_neis(node u, vector<node> &res) const {
    vector<bool> seen(vertex_num, false);
    res.clear();
    res.reserve(vertex_num);
    size_t u_start_ptr = csr_vlist[u], u_end_ptr = csr_vlist[u+1];
    for (size_t i = u_start_ptr; i < u_end_ptr; i++) {
        node v = csr_elist[i];
        size_t v_start_ptr = csr_vlist[v], v_end_ptr = csr_vlist[v+1];
        for (size_t j = v_start_ptr; j < v_end_ptr; j++) {
            node w = csr_elist[j];
            if (!seen[w]) {
                res.push_back(w);
                seen[w] = true;
            }
        }
    }
    res.shrink_to_fit();
}

void Graph::print_neis(node u) const {
    printf("N(%d):", u);
    for (int i = csr_vlist[u]; i < csr_vlist[u+1]; i++) {
        printf(" %d", csr_elist[i]);
    }
    printf(".\n");
}

void Graph::reorder(const vector<node> &origin_to_new){
	vector<bool> seen(vertex_num, false);
	vector<vector<node>> adj_list(vertex_num);
	for(size_t i = 0; i < vertex_num; ++i) {
		node new_id = origin_to_new[i];
		if (new_id == UINT_MAX) continue;
		if (seen[new_id]) printf("Error on node %lu! Already has node with label %d!\n", i, new_id);
		else seen[new_id] = true;
        for (size_t j = csr_vlist[i]; j < csr_vlist[i+1]; ++j) {
            node v = csr_elist[j];
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
    output_csr_vlist_binfile.write(reinterpret_cast<const char*>(csr_vlist.data()), sizeof(int) * csr_vlist.size());
    printf("Size of csr_vlist: %lu.\n", csr_vlist.size());
    output_csr_vlist_binfile.close();

    ofstream output_csr_elist_binfile(csr_elist_binfile, ios::binary);
    output_csr_elist_binfile.write(reinterpret_cast<const char*>(csr_elist.data()), sizeof(node) * csr_elist.size());
    printf("Size of csr_elist: %lu.\n", csr_elist.size());
    output_csr_elist_binfile.close();
}

void Graph::save_as_ligra(const string ligrafile) const {
    ofstream output_ligrafile(ligrafile);
    output_ligrafile << "AdjacencyGraph" << endl;
    output_ligrafile << vertex_num << endl;
    output_ligrafile << edge_num << endl;
    for (node u = 0; u < vertex_num; u++) {
        output_ligrafile << csr_vlist[u] << endl;
    }
    for (size_t i = 0; i < csr_elist.size(); i++) {
        output_ligrafile << csr_elist[i] << endl;
    }
    output_ligrafile.close();
}

// Algorithms of Graph class
void Graph::URP(const size_t part_num, vector<node> &vertex2partition) {
    vertex2partition.clear();
    vertex2partition.resize(vertex_num);
    for (node u = 0; u < vertex_num; u++) {
        vertex2partition[u] = rand() % part_num;
    }
}

void Graph::GRP(const size_t partition_number, vector<int> &vertex2partition) {
    std::random_device rd;
    std::mt19937 gen(rd());
	vector<pair<node, int>> degree_list(Graph::vertex_num);
	vertex2partition.clear();
    vertex2partition.resize(vertex_num, -1);
	for (size_t i = 0; i < Graph::vertex_num; i += 1) {
		degree_list[i] = make_pair((node)i, Graph::csr_vlist[i + 1] - Graph::csr_vlist[i]);
	}
	sort(degree_list.begin(), degree_list.end(), [](pair<node, int> a, pair<node, int> b) {
		return a.second > b.second;
		});
    double _lb = degree_list[0].second / (double)partition_number;
    printf("Lower bound: %lf\n", _lb);
	for (pair<node, int> item : degree_list) {
		vector<int> count_p_neigh(partition_number, 0);
		vector<node> not_selected;
        node u = item.first;
		for (size_t i = csr_vlist[u]; i < csr_vlist[u + 1]; i++) {
			node cur_v = csr_elist[i];
			int v_p = vertex2partition[cur_v];
			if (v_p > -1) {
				count_p_neigh[v_p]++;
			}
			else {
				not_selected.push_back(cur_v);
			}
		}

		if (not_selected.size()) {
            double expected_part_deg = (double) item.second / (double) partition_number;
            shuffle(not_selected.begin(), not_selected.end(), gen);
            vector<double> p_list(partition_number, 0);
            for (size_t i = 0; i < partition_number; i++) {
                p_list[i] = max(0.0, expected_part_deg - (double) count_p_neigh[i]);
            }
            double p_list_sum = accumulate(p_list.begin(), p_list.end(), 0.0);
            size_t ptr = 0;
            for (size_t i = 0; i < partition_number; i++) {
                size_t len = (size_t) floor(p_list[i] / p_list_sum * not_selected.size());
                for (size_t j = 0; j < len; j++) {
                    vertex2partition[not_selected[ptr++]] = i;
                }
            }
            if (ptr < not_selected.size()) {
                for (size_t i = ptr; i < not_selected.size(); i++) {
                    vertex2partition[not_selected[i]] = rand() % partition_number;
                }
            }
		}
	}
    for (size_t i = 0; i < vertex_num; i++) {
        if (vertex2partition[i] == -1) {
            vertex2partition[i] = rand() % partition_number;
        }
    }
}

void Graph::FGP(const size_t partition_num, vector<int> &vertex2partition) {
    vector<vector<int>> partition_neis_counts(partition_num, vector<int> (vertex_num, 0));

    vector<pair<node, int>> degree_list(Graph::vertex_num);
	vertex2partition.clear();
	for (size_t i = 0; i < Graph::vertex_num; i += 1) {
		vertex2partition.push_back(-1);
		degree_list[i] = make_pair((node)i, Graph::csr_vlist[i + 1] - Graph::csr_vlist[i]);
	}
	sort(degree_list.begin(), degree_list.end(), [](pair<node, int> a, pair<node, int> b) {
		return a.second > b.second;
		});
    double _lb = degree_list[0].second / (double) partition_num;
    printf("Lower bound: %lf\n", _lb);

    for (pair<node, int> item : degree_list) {
        node u = item.first;
        // printf("-- Vertex: %d, degree: %d.\n", u, item.second);
        int u_part = -1;
        if (csr_vlist[u+1] - csr_vlist[u] > 0) {
            size_t minimax_part_neis = vertex_num;
            for (size_t k = 0; k < partition_num; k++) {
                size_t max_curr_part_neis = 0;
                for (size_t j = csr_vlist[u]; j < csr_vlist[u+1]; j++) {
                    node v = csr_elist[j];
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
            node v = csr_elist[j];
            partition_neis_counts[u_part][v]++;
        }
    }
}