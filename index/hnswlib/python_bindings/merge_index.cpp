#include <vector>
#include <string>

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
    size_t random_seed
) {
    Index<dist_t>* merged_index = new Index<dist_t>(space_name, dim);
    merged_index->init_new_index(total_max_elements, M, efConstruction, random_seed, false);
    
    for (const auto& path : filenames) {
        // Load the split index
        // We assume 0 for max_elements works for loading (uses file header)
        hnswlib::HierarchicalNSW<dist_t> part_alg(merged_index->l2space, path, false, 0);
        
        // Iterate over all elements
        // label_lookup_ is a map<labeltype, tableint>
        for (const auto& kv : part_alg.label_lookup_) {
            hnswlib::labeltype label = kv.first;
            // getDataByLabel returns std::vector<dist_t>
            std::vector<dist_t> data = part_alg.template getDataByLabel<dist_t>(label);
            
            // Add to merged index
            merged_index->appr_alg->addPoint((void*)data.data(), label);
        }
    }
    
    merged_index->cur_l = merged_index->appr_alg->cur_element_count;
    return merged_index;
}
