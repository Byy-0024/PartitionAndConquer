#include "PartitionedGraph.hpp"

using namespace std;

Partitioned_Graph::Partitioned_Graph(const Graph &G, const vector<node> &_vertex2partition) {
    vertex2partition = _vertex2partition;
    vertex_num = G.get_vertex_num();
    edge_num = G.get_edge_num();
    partition_num = *max_element(vertex2partition.begin(), vertex2partition.end()) + 1;
    thread_num = partition_num;
    partition_cnts.resize(partition_num);
    for (size_t i = 0; i < vertex_num; i++) {
        // if (vertex2partition[i] < 0) printf("vertex2partition[%d] = %d\n", (int)i, vertex2partition[i]);
        partition_cnts[vertex2partition[i]] ++;
    }
    csr_vlist.resize(vertex_num * partition_num + 1);
    csr_elist.resize(edge_num);
    size_t min_partition_edge_num = edge_num, max_partition_edge_num = 0;
    vector<vector<vector<node>>> partitioned_adj_list(vertex_num, vector<vector<node>>(partition_num));
    vector<node> neis;
    for (node u = 0; u < vertex_num; u++) {
        G.get_neis(u, neis);
        for (auto v : neis) {
            partitioned_adj_list[u][vertex2partition[v]].push_back(v);
        }
    }
    for (size_t k = 0; k < partition_num; k++) {
        for (node u = 0; u < vertex_num; u++) {
            csr_vlist[k * vertex_num + u] = csr_elist.size();
            sort(partitioned_adj_list[u][k].begin(), partitioned_adj_list[u][k].end());
            csr_elist.insert(csr_elist.end(), partitioned_adj_list[u][k].begin(), partitioned_adj_list[u][k].end());
        }
        size_t partition_edge_num = csr_elist.size() - csr_vlist[k * vertex_num];
        if (partition_edge_num < min_partition_edge_num) min_partition_edge_num = partition_edge_num;
        if (partition_edge_num > max_partition_edge_num) max_partition_edge_num = partition_edge_num;
    }
    csr_vlist[vertex_num * partition_num] = csr_elist.size();
    // printf("Max partition edge num: %zu, min partition edge num: %zu, imbalance ratio: %.3f.\n", max_partition_edge_num, min_partition_edge_num, (double) max_partition_edge_num / (double) min_partition_edge_num);
}

// Operations of Partitioned_Graph class
size_t Partitioned_Graph::cnt_common_neis_galloping(const node u, const node v, const int partition_id) const {
    size_t cnt = 0;
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    size_t v_ptr = csr_vlist[partition_id * vertex_num + v], v_end_ptr = csr_vlist[partition_id * vertex_num + v + 1];
    if (u_end_ptr - u_ptr > v_end_ptr - v_ptr) { 
        swap(u_ptr, v_ptr);
        swap(u_end_ptr, v_end_ptr);
    }
    for (size_t i = u_ptr; i < u_end_ptr; i++) {
        node target = csr_elist[i];
        size_t offset = lower_bound(csr_elist.begin() + v_ptr, csr_elist.begin() + v_end_ptr, target) - csr_elist.begin();
        if (offset < v_end_ptr && csr_elist[offset] == target) cnt++;
    }
    return cnt;
}

size_t Partitioned_Graph::cnt_common_neis_galloping(const vector<node> &q, const int partition_id) const {
    size_t cnt = 0, min_deg = vertex_num;
    node u;
    for (auto v : q) {
        size_t deg_v = csr_vlist[partition_id * vertex_num + v + 1] - csr_vlist[partition_id * vertex_num + v];
        if (deg_v < min_deg) {
            min_deg = deg_v;
            u = v;
        }
    }
    size_t u_ptr = csr_vlist[partition_id * vertex_num + u], u_end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    for (size_t i = u_ptr; i < u_end_ptr; i++) {
        node target = csr_elist[i];
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

size_t Partitioned_Graph::cnt_common_neis_merge(const vector<node> &q, const int partition_id) const {
    vector<node> tmp_res1, tmp_res2;
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
    vector<node>().swap(tmp_res1);
    vector<node>().swap(tmp_res2);
    return res;
}

size_t Partitioned_Graph::cnt_common_neis_merge(const node u, const node v, const int partition_id) const {
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

size_t Partitioned_Graph::cnt_common_neis_merge(const node u, const vector<node> &nodes, const int partition_id) const {
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

bool value_greater(const pair<node, node> &a, const pair<node, node> &b) {
    return a.second > b.second;
}

size_t Partitioned_Graph::get_edge_num() const {
    return edge_num;
}

size_t Partitioned_Graph::get_partition_num() const {
    return partition_num;
}

size_t Partitioned_Graph::get_vertex_num() const {
    return vertex_num;
}

void Partitioned_Graph::evaluate_partition() const {
    vector<int> degs(vertex_num, 0);
    int max_part_deg = 0; 
    for (int i = 0; i < partition_num; i++) {
        for (int j = 0; j < vertex_num; j++) {
            int part_deg = csr_vlist[i * vertex_num + j + 1] - csr_vlist[i * vertex_num + j];
            degs[j] += part_deg;
            max_part_deg = max(max_part_deg, part_deg);
        }
    }
    int max_deg = 0;
    for (int i = 0; i < vertex_num; i++) {
        max_deg = max(max_deg, degs[i]);
    }
    double approximate_error = (double) (partition_num * max_part_deg) / (double) max_deg - 1.0;
    printf("max part deg: %d, max deg: %d, approximate error: %.3f%%.\n", max_part_deg, max_deg, approximate_error * 100.0);
}

void Partitioned_Graph::get_common_neis(const node u, const vector<node> &nodes, const int partition_id, vector<node>& res) const {
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

void Partitioned_Graph::get_neis(const node u, const int partition_id, vector<node> &res) const {
    size_t start_ptr = csr_vlist[partition_id * vertex_num + u], end_ptr = csr_vlist[partition_id * vertex_num + u + 1];
    res.assign(csr_elist.begin() + start_ptr, csr_elist.begin() + end_ptr);
}

void Partitioned_Graph::reorder() {
    vector<node> origin2new(vertex_num);
    vector<node> new2origin(vertex_num);
    vector<vector<node>> partition_list(partition_num);
    for (node u = 0; u < vertex_num; u++) {
        partition_list[vertex2partition[u]].push_back(u);
    }
    // Obtain the new vertex ordering.
    node curr_new_id = 0;
    for (int i = 0; i < partition_num; i++) {
        for (node u : partition_list[i]) {
            new2origin[curr_new_id] = u;
            origin2new[u] = curr_new_id++;
        }
    }

    // Reorder the CSR according to the obtained ordering.
    vector<node> csr_vlist_new(partition_num * vertex_num + 1);
    vector<node> csr_elist_new;
    csr_elist_new.reserve(csr_elist.size());
    for (int i = 0; i < partition_num; i++) {
        for (node u = 0; u < vertex_num; u++) {
            csr_vlist_new[i * vertex_num + u] = csr_elist_new.size();
            node u_origin_id = new2origin[u];
            for (size_t j = csr_vlist[i * vertex_num + u_origin_id]; j < csr_vlist[i * vertex_num + u_origin_id + 1]; j++) {
                csr_elist_new.push_back(origin2new[csr_elist[j]]);
            }
        }
    }
    csr_vlist_new[partition_num * vertex_num] = csr_elist_new.size();
    csr_vlist.swap(csr_vlist_new);
    csr_elist.swap(csr_elist_new);

    // Free space.
    vector<node>().swap(csr_vlist_new);
    vector<node>().swap(csr_elist_new);
}

// Algorithms of Partitioned_Graph class
size_t Partitioned_Graph::solve_triangle_counting_galloping() const {
    // vector<size_t> cnts(partition_num, 0);
    size_t res = 0;
    omp_set_num_threads(thread_num);
#pragma omp parallel for reduction(+: res)
    for (size_t i = 0; i < partition_num; i++) {
        for (size_t j = 0; j < partition_num; j++) {
            for (node u = 0; u < vertex_num; u++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    node v = csr_elist[k];
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
    omp_set_num_threads(thread_num);
#pragma omp parallel for reduction(+: res)
    for (size_t i = 0; i < partition_num; i++) {
        for (size_t j = 0; j < partition_num; j++) {
            for (node u = 0; u < vertex_num; u++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    node v = csr_elist[k];
                    // size_t local_cnt = cnt_common_neis_merge(u, v, i);
                    // res += local_cnt;
                    res += cnt_common_neis_merge(u, v, i);
                }
            }
        }
    }
    return res;
} 

void Partitioned_Graph::solve_cdlp(vector<int> &labels, int max_iteration) const {
#pragma omp declare reduction(vec_max : vector<int> : \
        std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), [](int a, int b) {return max(a, b);}) ) \
        initializer(omp_priv = vector<int>(omp_orig.size(), 0))
    
    for (int iter = 0; iter < max_iteration; iter++) {
        vector<int> tmp_labels(vertex_num, 0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_max : tmp_labels)
        for (int i = 0; i < partition_num; i++) {
            for (node u = 0; u < vertex_num; u++) {
                if (csr_vlist[i * vertex_num + u + 1] == csr_vlist[i * vertex_num + u]) continue;
                for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++)
                    tmp_labels[u] = max(tmp_labels[u], labels[csr_elist[j]]);
            }
        }
        // if (tmp_labels == labels) break;
        labels.swap (tmp_labels);
    }
}

void Partitioned_Graph::solve_jsim_galloping(const std::vector<node> &queries, std::vector<double> &res) const {
    const int n = queries.size();
    res.assign(n * n, 0.0);
#pragma omp declare reduction(vec_add : vector<double> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) ) \
    initializer(omp_priv = vector<double>(omp_orig.size(), 0.0))

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (node u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num) reduction(vec_add:res)
    for (int k = 0; k < partition_num; k++) {
        for (int i = 0; i < n; i++) {
            node u = queries[i];
            for (int j = 0; j < n; j++) {
                node v = queries[j];
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

void Partitioned_Graph::solve_jsim_merge(const std::vector<node> &queries, std::vector<double> &res) const {
    const int n = queries.size();
    res.assign(n * n, 0.0);
#pragma omp declare reduction(vec_add : vector<double> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>()) ) \
    initializer(omp_priv = vector<double>(omp_orig.size(), 0.0))

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (node u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num) reduction(vec_add:res)
    for (int k = 0; k < partition_num; k++) {
        for (int i = 0; i < n; i++) {
            node u = queries[i];
            for (int j = 0; j < n; j++) {
                node v = queries[j];
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
        for (node u = 0; u < vertex_num; u++) {
            for (int j = 0; j < partition_num; j++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    node v = csr_elist[k];
                    res[u] += cnt_common_neis_galloping(u, v, i);
                }
            }
        }
    }

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (node u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num)
    for (node u = 0; u < vertex_num; u++) {
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
        for (node u = 0; u < vertex_num; u++) {
            for (int j = 0; j < partition_num; j++) {
                for (size_t k = csr_vlist[j * vertex_num + u]; k < csr_vlist[j * vertex_num + u + 1]; k++) {
                    node v = csr_elist[k];
                    res[u] += cnt_common_neis_merge(u, v, i);
                }
            }
        }
    }

    vector<double> degs(vertex_num, 0.0);
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:degs)
    for (int i = 0; i < partition_num; i++) {
        for (node u = 0; u < vertex_num; u++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
    }

#pragma omp parallel for num_threads(thread_num)
    for (node u = 0; u < vertex_num; u++) {
        if (degs[u] < 0.5) continue;
        res[u] /= (degs[u] * (degs[u] - 1));
    }
}

void Partitioned_Graph::solve_page_rank(vector<float> &pr_vals, float damping_factor, int num_iterations) const {
    vector<float> pr_vals_new(vertex_num);
    vector<float> degs(vertex_num, 0.0);
    float _damping = (1 - damping_factor) / vertex_num;
#pragma omp declare reduction(vec_add : vector<float> : \
    std::transform( omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>()) ) \
    initializer(omp_priv = vector<float>(omp_orig.size(), 0.0))

#pragma omp parallel for num_threads(thread_num)
    for (node u = 0; u < vertex_num; u++) {
        for (int i = 0; i < partition_num; i++) {
            degs[u] += csr_vlist[i * vertex_num + u + 1] - csr_vlist[i * vertex_num + u];
        }
        degs[u] /= damping_factor;
    }
    for (int iter = 0; iter < num_iterations; iter++) {
#pragma omp parallel for num_threads(thread_num)
        for (node u = 0; u < vertex_num; u++) {
            pr_vals_new[u] = _damping;
            pr_vals[u] /= degs[u]; 
        }
#pragma omp parallel for num_threads(thread_num) reduction(vec_add:pr_vals_new)
        for (int i = 0; i < partition_num; i++)
        {
            for (node u = 0; u < vertex_num; u++) {
                float _local_updata = 0.0;
                for (size_t j = csr_vlist[i * vertex_num + u]; j < csr_vlist[i * vertex_num + u + 1]; j++) {
                    node v = csr_elist[j];
                    pr_vals_new[u] += pr_vals[v];
                }
            }
        }
        swap(pr_vals, pr_vals_new);
    }
}