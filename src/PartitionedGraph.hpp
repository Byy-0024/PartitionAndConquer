#ifndef PARTITIONEDGRAPH_HPP
#define PARTITIONEDGRAPH_HPP

#include "Graph.hpp"

class Partitioned_Graph {
    public:
        Partitioned_Graph(const Graph &g, const std::vector<int> &_vertex2partition);

        // Operations
        size_t cnt_common_neis_galloping(const node u, const node v, const int partition_id) const;
        size_t cnt_common_neis_galloping(const std::vector<node> &q, const int partition_id) const;
        size_t cnt_common_neis_merge(const node u, const node v, const int partition_id) const;
        size_t cnt_common_neis_merge(const std::vector<node> &q, const int partition_id) const;
        size_t cnt_common_neis_merge(const node u, const std::vector<node> &nodes, const int partition_id) const;
        size_t get_vertex_num() const;
        size_t get_edge_num() const;
        size_t get_partition_num() const;
        void evaluate_partition() const;
        void get_common_neis(const node u, const std::vector<node> &nodes, const int partition_id, std::vector<node> &res) const;
        void get_neis(const node u, const int partition_id, std::vector<node> &res) const;
        void reorder();

        // Algorithms
        size_t solve_triangle_counting_galloping() const;
        size_t solve_triangle_counting_merge() const;
        void solve_bfs(node u, std::vector<node> &parents) const;
        void solve_cdlp(std::vector<int> &labels, int max_iterations) const;
        void solve_jsim_galloping(const std::vector<node> &queries, std::vector<double> &res) const;
        void solve_jsim_merge(const std::vector<node> &queries, std::vector<double> &res) const;
        void solve_lcc_galloping(std::vector<double> &res) const;
        void solve_lcc_merge(std::vector<double> &res) const;
        void solve_page_rank(std::vector<float> &pr_vals, float damping_factor, int num_iterations) const;

    private:
        size_t vertex_num, edge_num, partition_num, thread_num;
        std::vector<int> csr_vlist;
        std::vector<node> csr_elist;
        std::vector<node> vertex2partition;
        std::vector<size_t> partition_cnts;
};

#endif // PARTITIONEDGRAPH_HPP