#ifndef GRAPH_HPP
#define GRAPH_HPP
#pragma once
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <queue>
#include <random>
#include <stdio.h>
#include <string>
#include <string.h>
#include <unordered_set>
#include <vector>

typedef int VID;
typedef unsigned long long EID;
typedef int Weight;
typedef unsigned long long ull;

struct time_counter {
    // std::chrono::_V2::steady_clock::time_point t_start, t_end;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
    ull t_cnt = 0;
    void start();
    void stop();
    void print(std::string s);
    void clean();
};

class Graph {
    public:
        Graph(const std::string graphfile);
        Graph(const std::string csrdir, bool is_weighted);
        Graph(const std::vector<std::vector<VID>> &adj_list);
        
        // Operators.
        const EID* get_csr_vlist() const;
        const VID* get_csr_elist() const;
        const Weight* get_csr_weight() const;
        bool has_edge(VID u, VID v) const;
        bool has_directed_edge(VID u, VID v) const;
        bool is_weighted() const;
        EID get_edge_num() const;
        VID get_vertex_num() const;
        VID get_nei(VID u, EID offset) const;
        std::pair<VID, VID> get_edge(EID offset) const;
        size_t cnt_common_neis_galloping(VID u, VID v) const;
        size_t cnt_common_neis_galloping(const std::vector<VID> &q) const;
        size_t cnt_common_neis_merge(VID u, VID v) const;
        size_t cnt_common_neis_merge(VID u, const std::vector<VID> &nodes) const;
        size_t cnt_common_neis_merge(const std::vector<VID> &q) const;
        size_t cnt_common_neis_less_than_merge(VID u, VID v, VID threshold) const;
        size_t get_deg(VID u) const;
        size_t get_max_deg() const;
        size_t solve_friend_triangle_galloping(const VID root, const std::vector<VID> &nodes, const int num_threads) const;
        size_t solve_friend_triangle_merge(const VID root, const std::vector<VID> &nodes, const int num_threads) const;
        size_t solve_triangle_counting_galloping(const int num_threads) const;
        size_t solve_triangle_counting_merge(const int num_threads) const;
        ull size_in_bytes() const;
        void batch_cnt_common_neis_galloping(const std::vector<std::pair<VID, VID>> &queries, std::vector<size_t> &res, int num_threads) const;
        void batch_cnt_common_neis_galloping(const std::vector<std::vector<VID>> &queries, std::vector<size_t> &res, int num_threads) const;
        void batch_cnt_common_neis_merge(const std::vector<std::pair<VID, VID>> &queries, std::vector<size_t> &res, int num_threads) const;
        void batch_cnt_common_neis_merge(const std::vector<std::vector<VID>> &queries, std::vector<size_t> &res, int num_threads) const;
        void bfs(VID source, std::vector<VID> &parents) const;
        // void cdlp(std::vector<int> &labels, int num_labels, int max_iterations, int num_threads) const;
        void get_common_neis(VID u, VID v, std::vector<VID>& res) const;
	    void get_common_neis(VID u, const std::vector<VID> &nodes, std::vector<VID> &res) const;
        void get_degs(std::vector<int> &degs) const;
        void get_neis(VID u, std::vector<VID> &neis) const;
        void get_two_hop_neis(VID u, std::vector<VID> &res) const;
        void greedy_partition(const size_t partition_number, std::vector<int> &vertex2partition);
        void greedy_partition_2(const size_t partition_number, std::vector<int> &vertex2partition);
        void page_rank(std::vector<float> &pr_vals, float damping_factor, int num_iterations, int num_threads) const;
        void print_neis(VID u) const;
        void random_partition(const size_t partition_num, std::vector<int> &vertex2partition);
        void reorder(const std::vector<VID> &origin2new);
        void save_as_csr_binfile(const std::string csr_vlist_binfile, const std::string csr_elist_binfile);
        void save_as_ligra(const std::string ligrafile) const;
        void solve_bfs(VID u, std::vector<VID> &parents) const;
        void solve_cdlp(std::vector<int> &labels, int max_iterations, int num_threads) const;
        void solve_kcore(std::vector<int> &core_numbers, int num_threads) const;
        void solve_mis(std::vector<int> &res, int num_threads) const;
        void solve_jsim_galloping(const std::vector<VID> &queries, std::vector<double> &res, int num_threads) const;
        void solve_jsim_merge(const std::vector<VID> &queries, std::vector<double> &res, int num_threads) const;
        void solve_lcc_galloping(std::vector<double> &res, int num_threads) const;
        void solve_lcc_merge(std::vector<double> &res, int num_threads) const;
        void solve_sssp(std::vector<int> &dist, VID src) const;
        void sim_rank(std::vector<double> &res, const int num_iteration, const int num_threads);

    private:
        size_t vertex_num, edge_num;
        std::vector<EID> csr_vlist;
        std::vector<VID> csr_elist;
        std::vector<Weight> csr_weight;
};

class Partitioned_Graph {
    public:
        Partitioned_Graph(const Graph &g, const std::vector<int> &_vertex2partition, const int thread_num);

        size_t cnt_common_neis_galloping(const VID u, const VID v, const int partition_id) const;
        size_t cnt_common_neis_galloping(const std::vector<VID> &q, const int partition_id) const;
        size_t cnt_common_neis_less_than_merge(const VID u, const VID v, const VID threshold, const int partition_id) const;
        size_t cnt_common_neis_merge(const VID u, const VID v, const int partition_id) const;
        size_t cnt_common_neis_merge(const std::vector<VID> &q, const int partition_id) const;
        size_t cnt_common_neis_merge(const VID u, const std::vector<VID> &nodes, const int partition_id) const;
        size_t get_vertex_num() const;
        size_t get_edge_num() const;
        size_t get_partition_num() const;
        size_t solve_friend_triangle_galloping(const VID root, const std::vector<VID> &nodes) const;
        size_t solve_friend_triangle_merge(const VID root, const std::vector<VID> &nodes) const;
        size_t solve_square_counting() const;
        size_t solve_triangle_counting_galloping() const;
        size_t solve_triangle_counting_merge() const;
        // std::pair<int, int> evaluate_partition() const;
        void batch_cnt_common_neis_galloping(const std::vector<std::pair<VID, VID>> &queries, std::vector<size_t> &res) const;
        void batch_cnt_common_neis_galloping(const std::vector<std::vector<VID>> &queries, std::vector<size_t> &res) const;
        void batch_cnt_common_neis_galloping_fine_grained(const std::vector<std::pair<VID, VID>> &queries, std::vector<size_t> &res) const;
        void batch_cnt_common_neis_merge(const std::vector<std::pair<VID, VID>> &queries, std::vector<size_t> &res) const;
        void batch_cnt_common_neis_merge(const std::vector<std::vector<VID>> &queries, std::vector<size_t> &res) const;
        void batch_cnt_common_neis_merge_fine_grained(const std::vector<std::pair<VID, VID>> &queries, std::vector<size_t> &res) const;
        // void cdlp(std::vector<int> &labels, int num_labels, int max_iteration) const;
        void evaluate_partition() const;
        void get_common_neis(const VID u, const std::vector<VID> &nodes, const int partition_id, std::vector<VID> &res) const;
        void get_neis(const VID u, const int partition_id, std::vector<VID> &res) const;
        void page_rank(std::vector<float> &pr_vals, float damping_factor, int num_iterations) const;
        void page_rank_residual(std::vector<float> &pr_vals, float damping_factor, int num_iterations) const;
        void reorder();
        void solve_bfs(VID u, std::vector<VID> &parents) const;
        void solve_cdlp(std::vector<int> &labels, int max_iterations) const;
        void solve_kcore(std::vector<int> &core_numbers) const;
        void solve_mis(std::vector<int> &res) const;
        void solve_jsim_galloping(const std::vector<VID> &queries, std::vector<double> &res) const;
        void solve_jsim_merge(const std::vector<VID> &queries, std::vector<double> &res) const;
        void solve_lcc_galloping(std::vector<double> &res) const;
        void solve_lcc_merge(std::vector<double> &res) const;
        void solve_sssp(VID src, std::vector<int> &dist) const;

    private:
        size_t vertex_num, edge_num, partition_num, thread_num;
        std::vector<EID> csr_vlist;
        std::vector<VID> csr_elist;
        std::vector<Weight> csr_weight;
        std::vector<VID> vertex2partition;
        std::vector<EID> poffsets;
        std::vector<VID> plist;
};
#endif