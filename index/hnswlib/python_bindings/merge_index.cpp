#include <vector>
#include <string>

// We assume this file is included in bindings.cpp after Index class definition
// so we don't need to include hnswlib.h again or define Index class.

#ifdef _OPENMP
#include <omp.h>
#endif


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
        std::cerr << "start to merge index: " << path << std::endl;

        hnswlib::HierarchicalNSW<dist_t> part_alg(merged_index->l2space, path, false, 0);
        std::vector<hnswlib::labeltype> labels;
        labels.reserve(part_alg.label_lookup_.size());
        for (const auto& kv : part_alg.label_lookup_) {
            labels.push_back(kv.first);
        }

        std::cerr << "index size: " << labels.size() << std::endl;

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < labels.size(); ++i) {
            const hnswlib::labeltype label = labels[i];
            std::vector<dist_t> data = part_alg.template getDataByLabel<dist_t>(label);
            merged_index->appr_alg->addPoint((void*)data.data(), label);
        }
    }
    
    merged_index->cur_l = merged_index->appr_alg->cur_element_count;
    return merged_index;
}
