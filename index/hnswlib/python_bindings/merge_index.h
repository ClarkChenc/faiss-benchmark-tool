#pragma once

#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cstring>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// We assume this file is included in bindings.cpp after Index class definition
// so we don't need to include hnswlib.h again or define Index class.

class SameSegmentFilter : public hnswlib::BaseFilterFunctor {
    size_t start_id;
    size_t end_id;
 public:
    SameSegmentFilter(size_t start, size_t end) : start_id(start), end_id(end) {}
    bool operator()(hnswlib::labeltype id) override {
        return id < start_id || id >= end_id;
    }
};

template<typename dist_t>
Index<dist_t>* merge_indices(
    const std::vector<std::string>& filenames,
    const std::string& space_name,
    int dim,
    size_t total_max_elements,
    size_t M,
    size_t efConstruction,
    size_t random_seed,
    float ratio = 1.0f,
    float keep_pruned_connections = 1.0f
) {
    // 1. Initialize Merged Index Wrapper
    Index<dist_t>* merged_index_wrapper = new Index<dist_t>(space_name, dim);
    merged_index_wrapper->init_new_index(total_max_elements, M, efConstruction, random_seed, false);
    hnswlib::HierarchicalNSW<dist_t>* out_alg = merged_index_wrapper->appr_alg;
    
    // 2. Load all segments
    std::vector<hnswlib::HierarchicalNSW<dist_t>*> segments;
    segments.reserve(filenames.size());
    std::vector<size_t> offsets;
    offsets.reserve(filenames.size());
    size_t current_offset = 0;
    
    try {
        for (const auto& path : filenames) {
            std::cerr << "Loading segment: " << path << std::endl;
            hnswlib::HierarchicalNSW<dist_t>* seg = new hnswlib::HierarchicalNSW<dist_t>(
                merged_index_wrapper->l2space, path, false, 0
            );
            // Build indegree map for refinement candidates (keep all nodes with indegree > 0)
            seg->buildIndegreeMap(keep_pruned_connections);
            segments.push_back(seg);
            offsets.push_back(current_offset);
            current_offset += seg->cur_element_count;
        }
    } catch (...) {
        for (auto seg : segments) delete seg;
        delete merged_index_wrapper;
        throw;
    }
    
    out_alg->cur_element_count = current_offset;
    
    // 3. Entry Point Selection
    int max_level = -1;
    for (auto seg : segments) {
        if (seg->maxlevel_ > max_level) max_level = seg->maxlevel_;
    }
    out_alg->maxlevel_ = max_level;
    
    std::vector<hnswlib::tableint> potential_eps;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i]->maxlevel_ == max_level) {
            if (segments[i]->enterpoint_node_ != (size_t)-1)
                potential_eps.push_back(segments[i]->enterpoint_node_ + offsets[i]);
        }
    }
    
    if (!potential_eps.empty()) {
        std::cerr << "potential_eps size: " << potential_eps.size() << std::endl;
        // Simple random selection
        size_t idx = 0;
        if (potential_eps.size() > 1) {
            idx = rand() % potential_eps.size(); 
        }
        out_alg->enterpoint_node_ = potential_eps[idx];
    } else {
        out_alg->enterpoint_node_ = -1;
    }
    std::cerr << "final entry_point: " << out_alg->enterpoint_node_ << std::endl;

    std::cerr << "Merging " << segments.size() << " segments, total elements: " << current_offset << std::endl;

    // 4. Merge Data and Links (Parallel Copy)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < segments.size(); ++i) {
        auto seg = segments[i];
        size_t offset = offsets[i];
        
        for (size_t j = 0; j < seg->cur_element_count; ++j) {
            size_t new_id = j + offset;
            
            // 4a. Element Level
            out_alg->element_levels_[new_id] = seg->element_levels_[j];
            
            // 4b. Level 0 Data (includes vector + links + label)
            char* src_data = seg->data_level0_memory_ + j * seg->size_data_per_element_;
            char* dst_data = out_alg->data_level0_memory_ + new_id * out_alg->size_data_per_element_;
            
            memcpy(dst_data, src_data, out_alg->size_data_per_element_);
            
            // Relink Level 0
            unsigned int* linklist0 = (unsigned int*)(dst_data + out_alg->offsetLevel0_);
            int size0 = *linklist0; 
            hnswlib::tableint* links0 = (hnswlib::tableint*)(linklist0 + 1);
            for (int k = 0; k < size0; ++k) {
                links0[k] += offset;
            }
            
            // 4c. Higher Levels
            if (seg->element_levels_[j] > 0) {
                 size_t alloc_size = seg->element_levels_[j] * out_alg->size_links_per_element_;
                 out_alg->linkLists_[new_id] = (char*)malloc(alloc_size);
                 if (!out_alg->linkLists_[new_id]) {
                     // Memory error handling in OMP is tricky, but let's assume enough memory
                     continue; 
                 }
                 
                 memcpy(out_alg->linkLists_[new_id], seg->linkLists_[j], alloc_size);
                 
                 // Relink
                 for (int level = 1; level <= seg->element_levels_[j]; ++level) {
                      unsigned int* linklist = (unsigned int*)(out_alg->linkLists_[new_id] + (level-1) * out_alg->size_links_per_element_);
                      int size = *linklist;
                      hnswlib::tableint* links = (hnswlib::tableint*)(linklist + 1);
                      for (int k = 0; k < size; ++k) {
                          links[k] += offset;
                      }
                 }
            } else {
                 out_alg->linkLists_[new_id] = nullptr;
            }
        }
    }
    
    // Sequential Label Population
    for (size_t i = 0; i < segments.size(); ++i) {
        auto seg = segments[i];
        size_t offset = offsets[i];
        for (size_t j = 0; j < seg->cur_element_count; ++j) {
            out_alg->label_lookup_[seg->getExternalLabel(j)] = j + offset;
        }
    }

    // Prepare boundaries for segment filtering
    std::vector<size_t> boundaries = offsets;
    boundaries.push_back(current_offset);

    // 5. Refinement
    if (ratio > 0) {
        auto refine_layer = [&](const std::vector<hnswlib::tableint>& nodes,
                                hnswlib::HierarchicalNSW<dist_t>& cand_index,
                                size_t K,
                                int level) {
            size_t max_links = level == 0 ? out_alg->maxM0_ : out_alg->maxM_;
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < nodes.size(); ++i) {
                hnswlib::tableint u_id = nodes[i];
                void* u_data = out_alg->getDataByInternalId(u_id);
                auto it = std::upper_bound(boundaries.begin(), boundaries.end(), (size_t)u_id);
                size_t seg_idx = std::distance(boundaries.begin(), it) - 1;
                SameSegmentFilter filter(boundaries[seg_idx], boundaries[seg_idx+1]);
                auto res = cand_index.searchKnnCloserFirst(u_data, K, &filter);
                std::vector<hnswlib::tableint> new_neighbors;
                for (auto& pr : res) {
                    hnswlib::tableint v_id = (hnswlib::tableint)pr.second;
                    if (v_id != u_id) new_neighbors.push_back(v_id);
                }
                unsigned int* linklist;
                hnswlib::tableint* links;
                if (level == 0) {
                    linklist = (unsigned int*)(out_alg->data_level0_memory_ + u_id * out_alg->size_data_per_element_ + out_alg->offsetLevel0_);
                    links = (hnswlib::tableint*)(linklist + 1);
                } else {
                    linklist = (unsigned int*)(out_alg->linkLists_[u_id] + (level-1) * out_alg->size_links_per_element_);
                    links = (hnswlib::tableint*)(linklist + 1);
                }
                if (level == 0) {
                    int cur_size = *linklist;
                    std::unordered_set<hnswlib::tableint> current_set;
                    for(int k=0; k<cur_size; ++k) current_set.insert(links[k]);
                    std::vector<hnswlib::tableint> filtered;
                    for (auto id : new_neighbors) {
                        if (current_set.find(id) == current_set.end()) filtered.push_back(id);
                    }
                    size_t K = (size_t)(ratio * 2 * out_alg->M_);
                    if (filtered.empty()) {
                        out_alg->l0_merge_neighbors_[u_id].clear();
                    } else {
                        std::priority_queue<std::pair<dist_t, hnswlib::tableint>, std::vector<std::pair<dist_t, hnswlib::tableint>>, typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;
                        for (auto id : filtered) {
                            void* neighbor_data = out_alg->getDataByInternalId(id);
                            dist_t dist = out_alg->fstdistfunc_(u_data, neighbor_data, out_alg->dist_func_param_);
                            top_candidates.emplace(dist, id);
                        }
                        out_alg->getNeighborsByHeuristic2(top_candidates, K);
                        std::vector<hnswlib::tableint> selected;
                        while (!top_candidates.empty()) {
                            selected.push_back(top_candidates.top().second);
                            top_candidates.pop();
                        }
                        out_alg->l0_merge_neighbors_[u_id] = std::move(selected);
                    }
                } else {
                    int cur_size = *linklist;
                    std::unordered_set<hnswlib::tableint> candidate_ids;
                    for(int k=0; k<cur_size; ++k) candidate_ids.insert(links[k]);
                    for(auto id : new_neighbors) candidate_ids.insert(id);
                    if (candidate_ids.size() <= max_links) {
                        int k = 0;
                        for (auto id : candidate_ids) {
                            links[k++] = id;
                        }
                        *linklist = k;
                    } else {
                        std::priority_queue<std::pair<dist_t, hnswlib::tableint>, std::vector<std::pair<dist_t, hnswlib::tableint>>, typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst> top_candidates;
                        for (auto id : candidate_ids) {
                            void* neighbor_data = out_alg->getDataByInternalId(id);
                            dist_t dist = out_alg->fstdistfunc_(u_data, neighbor_data, out_alg->dist_func_param_);
                            top_candidates.emplace(dist, id);
                        }
                        out_alg->getNeighborsByHeuristic2(top_candidates, max_links);
                        int k = 0;
                        while (!top_candidates.empty()) {
                            links[k++] = top_candidates.top().second;
                            top_candidates.pop();
                        }
                        *linklist = k;
                    }
                }
            }
        };
        for (int level = max_level; level >= 1; --level) {
            std::vector<hnswlib::tableint> candidates;
            for (size_t i = 0; i < current_offset; ++i) {
                if (out_alg->element_levels_[i] >= level) {
                    candidates.push_back(i);
                }
            }
            if (candidates.empty()) continue;
            std::cerr << "Refining Level " << level << ": " << candidates.size() << " nodes with ratio " << ratio << std::endl;
            auto t_build_start_lvl = std::chrono::steady_clock::now();
            hnswlib::HierarchicalNSW<dist_t> cand_index(merged_index_wrapper->l2space, candidates.size(), M, efConstruction, random_seed);

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < candidates.size(); ++i) {
                void* p = out_alg->getDataByInternalId(candidates[i]);
                cand_index.addPoint(p, (hnswlib::labeltype)candidates[i], false);
            }
            cand_index.setEf(std::max((size_t)std::ceil(ratio * M) * 2, (size_t)10));
            auto t_build_end_lvl = std::chrono::steady_clock::now();
            std::cerr << "\tLevel " << level << " build tmp index time cost: "
                      << std::chrono::duration<double>(t_build_end_lvl - t_build_start_lvl).count() << "s" << std::endl;
            size_t K = (size_t)(ratio * M);
            refine_layer(candidates, cand_index, K, level);
        }
        std::vector<hnswlib::tableint> candidate_set;
        for (size_t i = 0; i < segments.size(); ++i) {
            size_t offset = offsets[i];
            for (const auto& kv : segments[i]->indegree_map_) {
                candidate_set.push_back(kv.first + offset);
            }
        }
        std::cerr << "Refining Level 0: " << candidate_set.size() << " candidates with ratio " << ratio << std::endl;
        if (!candidate_set.empty()) {
            auto t_build_start_l0 = std::chrono::steady_clock::now();
            hnswlib::HierarchicalNSW<dist_t> cand_index(merged_index_wrapper->l2space, candidate_set.size(), M, efConstruction, random_seed);

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < candidate_set.size(); ++i) {
                void* p = out_alg->getDataByInternalId(candidate_set[i]);
                cand_index.addPoint(p, (hnswlib::labeltype)candidate_set[i], false);
            }
            size_t K = (size_t)(ratio * 2 * M);
            cand_index.setEf(std::max(K * 2, (size_t)10));
            auto t_build_end_l0 = std::chrono::steady_clock::now();
            std::cerr << "\tLevel 0 build tmp index time cost: "
                      << std::chrono::duration<double>(t_build_end_l0 - t_build_start_l0).count() << "s" << std::endl;
            auto t_search_start_l0 = std::chrono::steady_clock::now();
            refine_layer(candidate_set, cand_index, K, 0);
            auto t_search_end_l0 = std::chrono::steady_clock::now();
            std::cerr << "\tLevel 0 candidate_set search cost: "
                      << std::chrono::duration<double>(t_search_end_l0 - t_search_start_l0).count() << "s" << std::endl;
        }
    }

    // Cleanup segments
    for (auto seg : segments) delete seg;
    
    return merged_index_wrapper;
}
