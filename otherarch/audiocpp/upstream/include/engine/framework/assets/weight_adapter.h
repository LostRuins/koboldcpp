#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <map>
#include <unordered_map>
#include <variant>

namespace engine::assets {

enum class WeightMergeMode {
    RankWiseF32,
    DeltaF32,
    RoundedBF16Delta,
};

struct LowRankWeightUpdate {
    std::vector<float> a;
    std::vector<float> b;
    int64_t r = 0;
    int64_t in = 0;
    int64_t out = 0;
};

struct KroneckerWeightUpdate {
    std::vector<float> left;
    std::vector<float> right;
    int64_t left_rows = 0;
    int64_t left_cols = 0;
    int64_t right_rows = 0;
    int64_t right_cols = 0;
};

struct WeightUpdate {
    std::variant<LowRankWeightUpdate, KroneckerWeightUpdate> factors;
    float scale = 1.0F;
    WeightMergeMode merge_mode = WeightMergeMode::DeltaF32;
};

struct WeightReplacement {
    std::vector<int64_t> shape;
    std::vector<float> values;
};

struct WeightAdapter {
    std::map<std::string, WeightUpdate> updates;
    std::map<std::string, WeightReplacement> replacements;
};

struct WeightAdapterReadOptions {
    float strength = 1.0F;
    // Overrides the effective alpha/r coefficient, not the user strength.
    std::optional<float> scale_override;
    bool require_scaling_metadata = false;
};

// Reads native A/B, PEFT A/B, ComfyUI up/down, and matrix LoKr factors.
// Target names remain external until bound to a model. No model names here.
WeightAdapter read_weight_adapter(
    const std::filesystem::path & path, const WeightAdapterReadOptions & options = {});

struct WeightAdapterTarget {
    std::string tensor;
    // Slice of the adapter's output rows, for fused external projections.
    int64_t first_row = 0;
    int64_t rows = 0;
};

class WeightAdapterBindings {
public:
    // Also registers the underscore spelling used by ComfyUI/LyCORIS.
    void add(std::string name, std::vector<WeightAdapterTarget> targets);
    void add(std::string name, std::string tensor);
    const std::vector<WeightAdapterTarget> & resolve(const std::string & name) const;
private:
    std::unordered_map<std::string, std::vector<WeightAdapterTarget>> targets_;
};

WeightAdapter bind_weight_adapter(
    const TensorSource & base, WeightAdapter adapter,
    const WeightAdapterBindings & bindings, WeightMergeMode merge_mode);

// Merges on host during weight loading, then uses the existing dtype/upload API.
// Inference graphs and base storage stay untouched. Replacements win over updates.
// Optional caching retains one uploaded dtype per adapted weight, never its variants.
std::shared_ptr<const TensorSource> make_weight_adapter_source(
    std::shared_ptr<const TensorSource> base, WeightAdapter adapter,
    std::string log_prefix = "adapter", bool cache_backend_weights = false);

}  // namespace engine::assets
