#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

// We assume this file is included in bindings.cpp after Index class definition
// so we don't need to include hnswlib.h again or define Index class.

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
        // Simple random selection
        size_t idx = 0;
        if (potential_eps.size() > 1) {
            idx = rand() % potential_eps.size(); 
        }
        out_alg->enterpoint_node_ = potential_eps[idx];
    } else {
        out_alg->enterpoint_node_ = -1;
    }

    std::cerr << "Merging " << segments.size() << " segments, total elements: " << current_offset << std::endl;

    // 4. Merge Data and Links (Parallel Copy)
    #pragma omp parallel for schedule(dynamic)
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

    // 5. Refinement (Layer 0)
    std::vector<hnswlib::tableint> candidate_set;
    for (size_t i = 0; i < segments.size(); ++i) {
        size_t offset = offsets[i];
        for (const auto& kv : segments[i]->indegree_map_) {
             candidate_set.push_back(kv.first + offset);
        }
    }

    std::cerr << "Refining " << candidate_set.size() << " candidates with ratio " << ratio << std::endl;

    if (!candidate_set.empty() && ratio > 0) {
        size_t K = (size_t)(ratio * 2 * M); 
        size_t max_links0 = out_alg->maxM0_;
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < candidate_set.size(); ++i) {
            hnswlib::tableint u_id = candidate_set[i];
            
            // Brute force search in candidate_set for K nearest
            // Priority queue stores pairs <distance, id>, ordered by distance (largest on top)
            std::priority_queue<std::pair<dist_t, hnswlib::tableint>> top_k;
            
            void* u_data = out_alg->getDataByInternalId(u_id);
            
            for (size_t j = 0; j < candidate_set.size(); ++j) {
                if (i == j) continue;
                hnswlib::tableint v_id = candidate_set[j];
                void* v_data = out_alg->getDataByInternalId(v_id);
                
                dist_t d = out_alg->fstdistfunc_(u_data, v_data, out_alg->dist_func_param_);
                
                top_k.push({d, v_id});
                if (top_k.size() > K) top_k.pop();
            }
            
            std::vector<hnswlib::tableint> new_neighbors;
            while(!top_k.empty()) {
                new_neighbors.push_back(top_k.top().second);
                top_k.pop();
            }
            // new_neighbors are from largest dist to smallest. 
            // Reverse to put smallest (best) first
            std::reverse(new_neighbors.begin(), new_neighbors.end());
            
            // Merge into u_id's link list
            unsigned int* linklist0 = (unsigned int*)(out_alg->data_level0_memory_ + u_id * out_alg->size_data_per_element_ + out_alg->offsetLevel0_);
            int cur_size = *linklist0;
            hnswlib::tableint* links0 = (hnswlib::tableint*)(linklist0 + 1);
            
            std::unordered_set<hnswlib::tableint> current_set;
            for(int k=0; k<cur_size; ++k) current_set.insert(links0[k]);
            
            std::vector<hnswlib::tableint> to_add;
            for (auto id : new_neighbors) {
                if (current_set.find(id) == current_set.end() && id != u_id) {
                    to_add.push_back(id);
                }
            }
            
            size_t needed = to_add.size();
            if (needed > 0) {
                if (cur_size + needed <= max_links0) {
                     for (auto id : to_add) {
                         links0[cur_size++] = id;
                     }
                     *linklist0 = cur_size;
                } else {
                     // Fill from back to front, prioritizing to_add
                     size_t start_idx = max_links0 - needed;
                     if (start_idx > cur_size) start_idx = cur_size; // Should cover if needed > max_links0 case too
                     
                     // If needed > max_links0, we just take the first max_links0 of to_add (which are best)
                     if (needed >= max_links0) {
                         start_idx = 0;
                         for(size_t k=0; k<max_links0; ++k) links0[k] = to_add[k];
                         *linklist0 = max_links0;
                     } else {
                         // Normal case: replace tail
                         for (size_t k = 0; k < needed; ++k) {
                             links0[start_idx + k] = to_add[k];
                         }
                         *linklist0 = max_links0;
                     }
                }
            }
        }
    }

    // Cleanup segments
    for (auto seg : segments) delete seg;
    
    return merged_index_wrapper;
}
