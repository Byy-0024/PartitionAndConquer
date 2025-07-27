#ifndef GRAPH_HPP
#define GRAPH_HPP
#pragma once
#include <algorithm>
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

typedef int node;
typedef unsigned long long ull;

struct time_counter {
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
        Graph(const std::string csr_vlist_binfile, const std::string csr_elist_binfile);
        Graph(const std::vector<std::vector<node>> &adj_list);
        
        // Operators.
        const int* get_csr_vlist() const;
        const node* get_csr_elist() const;
        bool has_edge(node u, node v) const;
        bool has_directed_edge(node u, node v) const;
        node get_nei(node u, size_t offset) const;
        std::pair<node, node> get_edge(size_t offset) const;
        size_t cnt_common_neis_galloping(node u, node v) const;
        size_t cnt_common_neis_galloping(const std::vector<node> &q) const;
        size_t cnt_common_neis_merge(node u, node v) const;
        size_t cnt_common_neis_merge(node u, const std::vector<node> &nodes) const;
        size_t get_edge_num() const;
        size_t get_vertex_num() const;
        size_t get_deg(node u) const;
        size_t get_max_deg() const;
        ull size_in_bytes() const;
        void get_common_neis(node u, node v, std::vector<node>& res) const;
	    void get_common_neis(node u, const std::vector<node> &nodes, std::vector<node> &res) const;
        void get_degs(std::vector<int> &degs) const;
        void get_neis(node u, std::vector<node> &neis) const;
        void get_two_hop_neis(node u, std::vector<node> &res) const;
        void print_neis(node u) const;
        void reorder(const std::vector<node> &origin2new);
        void save_as_csr_binfile(const std::string csr_vlist_binfile, const std::string csr_elist_binfile);
        void save_as_ligra(const std::string ligrafile) const;

        // Algorithms
        void GRP(const size_t partition_number, std::vector<int> &vertex2partition); // GRP: Greedy Random Partition.
        void FGP(const size_t partition_number, std::vector<int> &vertex2partition); // FGP: Fine-grained Greedy Partition.
        void URP(const size_t partition_num, std::vector<int> &vertex2partition);  // URP: Uniform Random Partition.

    private:
        size_t vertex_num, edge_num;
        std::vector<int> csr_vlist;
        std::vector<node> csr_elist;
};
#endif