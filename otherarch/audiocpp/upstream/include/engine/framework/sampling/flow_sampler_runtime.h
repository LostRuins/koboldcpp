#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::modules {

struct FlowSamplerScheduleStep {
    int64_t index = 0;
    float t = 0.0F;
    float t_next = 0.0F;
    float sigma = 0.0F;
    float sigma_next = 0.0F;
};

struct FlowSamplerBranchSpec {
    std::string name;
    float weight = 1.0F;
};

struct FlowSamplerCacheSpec {
    std::string name;
    std::string mode;
    bool reset_on_sequence_begin = true;
};

struct FlowSamplerCacheKey {
    std::string name;
    std::string mode;

    bool operator==(const FlowSamplerCacheKey & other) const;
    bool operator!=(const FlowSamplerCacheKey & other) const {
        return !(*this == other);
    }
};

struct FlowSamplerCacheState {
    std::string name;
    std::string mode;
    int64_t revision = 0;
    bool initialized = false;

    bool operator==(const FlowSamplerCacheState & other) const;
    bool operator!=(const FlowSamplerCacheState & other) const {
        return !(*this == other);
    }
};

enum class FlowSamplerCacheUpdateKind {
    Unchanged,
    Updated,
    Reset,
};

struct FlowSamplerCacheUpdate {
    std::string name;
    FlowSamplerCacheUpdateKind kind = FlowSamplerCacheUpdateKind::Unchanged;
};

enum class FlowSamplerPredictionType {
    Velocity,
};

enum class FlowSamplerUpdateRule {
    Euler,
    Custom,
};

enum class FlowSamplerGuidanceMode {
    None,
    ClassifierFree,
};

struct FlowSamplerGuidanceConfig {
    FlowSamplerGuidanceMode mode = FlowSamplerGuidanceMode::None;
    std::string cond_branch;
    std::string uncond_branch;
    float scale = 1.0F;
};

struct FlowSamplerPrefixPolicy {
    std::vector<float> preserve_mask;
};

struct FlowSamplerGraphKey {
    std::vector<int64_t> latent_shape;
    int64_t branch_count = 0;
    int64_t schedule_steps = 0;
    std::string sampler_mode;
    std::vector<FlowSamplerCacheKey> caches;
    int64_t modulation_revision = 0;

    bool operator==(const FlowSamplerGraphKey & other) const;
    bool operator!=(const FlowSamplerGraphKey & other) const {
        return !(*this == other);
    }
};

struct FlowSamplerRuntimeConfig {
    std::string label = "flow_sampler";
    std::vector<int64_t> latent_shape;
    std::vector<float> initial_latent;
    std::vector<FlowSamplerScheduleStep> schedule;
    std::vector<FlowSamplerBranchSpec> branches;
    std::vector<FlowSamplerCacheSpec> caches;
    FlowSamplerPredictionType prediction_type = FlowSamplerPredictionType::Velocity;
    FlowSamplerUpdateRule update_rule = FlowSamplerUpdateRule::Euler;
    FlowSamplerGuidanceConfig guidance;
    FlowSamplerPrefixPolicy prefix;
    bool release_graph_after_sequence = false;
    bool require_complete_graph_key = true;
};

struct FlowSamplerStepState {
    int64_t sequence_index = 0;
    FlowSamplerScheduleStep schedule;
    std::vector<FlowSamplerBranchSpec> branches;
    std::vector<FlowSamplerCacheState> caches;
    FlowSamplerGraphKey graph_key;
};

struct FlowSamplerSequenceState {
    std::vector<int64_t> latent_shape;
    std::vector<FlowSamplerScheduleStep> schedule;
    std::vector<FlowSamplerBranchSpec> branches;
    std::vector<FlowSamplerCacheState> caches;
};

struct FlowSamplerDenoiserInput {
    FlowSamplerStepState state;
    const std::vector<float> & latent;
};

struct FlowSamplerBranchPrediction {
    std::string branch;
    std::vector<float> values;
};

struct FlowSamplerDenoiserOutput {
    std::vector<FlowSamplerBranchPrediction> predictions;
    std::vector<FlowSamplerCacheUpdate> cache_updates;
};

struct FlowSamplerUpdateInput {
    FlowSamplerStepState state;
    const std::vector<float> & prediction;
    std::vector<float> & latent;
};

class FlowSamplerUpdateRuntime {
public:
    virtual ~FlowSamplerUpdateRuntime();

    virtual void update_latent(const FlowSamplerUpdateInput & input) = 0;
};

std::unique_ptr<FlowSamplerUpdateRuntime> make_flow_sampler_euler_update();

class FlowSamplerDenoiserRuntime {
public:
    virtual ~FlowSamplerDenoiserRuntime();

    virtual void reset_sampler_caches(const std::vector<FlowSamplerCacheState> & caches) = 0;
    virtual std::vector<FlowSamplerCacheUpdate> begin_sampler_sequence(const FlowSamplerSequenceState & state) = 0;
    virtual FlowSamplerGraphKey sampler_graph_key(const FlowSamplerStepState & state) = 0;
    virtual void rebuild_sampler_graph(
        const FlowSamplerGraphKey & key,
        const FlowSamplerStepState & state) = 0;
    virtual FlowSamplerDenoiserOutput run_sampler_denoiser(const FlowSamplerDenoiserInput & input) = 0;
    virtual void release_sampler_graphs() = 0;
};

struct FlowSamplerSingleBranchDenoiserConfig {
    std::string label = "flow_sampler.denoiser";
    std::string branch_name = "velocity";
    std::function<void(const std::vector<FlowSamplerCacheState> &)> reset_caches;
    std::function<std::vector<FlowSamplerCacheUpdate>(const FlowSamplerSequenceState &)> begin_sequence;
    std::function<std::string(const FlowSamplerStepState &)> sampler_mode;
    std::function<int64_t(const FlowSamplerStepState &)> modulation_revision;
    std::function<void(const FlowSamplerGraphKey &, const FlowSamplerStepState &)> rebuild_graph;
    std::function<std::vector<float>(const FlowSamplerDenoiserInput &)> predict;
    std::function<void()> release_graphs;
};

std::unique_ptr<FlowSamplerDenoiserRuntime>
make_flow_sampler_single_branch_denoiser(FlowSamplerSingleBranchDenoiserConfig config);

struct FlowSamplerSingleBranchDenoiserRuntimeConfig {
    std::string label = "flow_sampler.denoiser";
    std::string branch_name = "velocity";
};

class FlowSamplerSingleBranchDenoiserRuntime : public FlowSamplerDenoiserRuntime {
public:
    explicit FlowSamplerSingleBranchDenoiserRuntime(
        FlowSamplerSingleBranchDenoiserRuntimeConfig config);
    ~FlowSamplerSingleBranchDenoiserRuntime() override;

    void reset_sampler_caches(const std::vector<FlowSamplerCacheState> & caches) override;
    std::vector<FlowSamplerCacheUpdate> begin_sampler_sequence(
        const FlowSamplerSequenceState & state) override;
    FlowSamplerGraphKey sampler_graph_key(const FlowSamplerStepState & state) override;
    void rebuild_sampler_graph(
        const FlowSamplerGraphKey & key,
        const FlowSamplerStepState & state) override;
    FlowSamplerDenoiserOutput run_sampler_denoiser(const FlowSamplerDenoiserInput & input) override;
    void release_sampler_graphs() override;

protected:
    virtual void reset_sequence_caches(const std::vector<FlowSamplerCacheState> & caches);
    virtual std::vector<FlowSamplerCacheUpdate> begin_sequence(
        const FlowSamplerSequenceState & state);
    virtual std::string sampler_mode(const FlowSamplerStepState & state) const = 0;
    virtual int64_t modulation_revision(const FlowSamplerStepState & state) const;
    virtual void rebuild_graph(
        const FlowSamplerGraphKey & key,
        const FlowSamplerStepState & state) = 0;
    virtual std::vector<float> predict_branch(const FlowSamplerDenoiserInput & input) = 0;
    virtual void release_runtime_graphs();

    const FlowSamplerSingleBranchDenoiserRuntimeConfig & config() const noexcept;

private:
    FlowSamplerSingleBranchDenoiserRuntimeConfig config_;
    std::vector<int64_t> sequence_latent_shape_;
    int64_t sequence_schedule_steps_ = 0;
};

class FlowSamplerRuntime {
public:
    FlowSamplerRuntime(
        FlowSamplerRuntimeConfig config,
        std::unique_ptr<FlowSamplerDenoiserRuntime> denoiser);
    FlowSamplerRuntime(
        FlowSamplerRuntimeConfig config,
        std::unique_ptr<FlowSamplerDenoiserRuntime> denoiser,
        std::unique_ptr<FlowSamplerUpdateRuntime> updater);
    ~FlowSamplerRuntime();

    FlowSamplerRuntime(const FlowSamplerRuntime &) = delete;
    FlowSamplerRuntime & operator=(const FlowSamplerRuntime &) = delete;

    void run_sequence();
    void run_sequence(const std::vector<float> & initial_latent);
    void release_runtime_graphs();
    void reset_graph_reuse_state();
    void reset_runtime_caches();

    const std::vector<float> & latent() const noexcept;
    const FlowSamplerRuntimeConfig & config() const noexcept;
    const std::optional<FlowSamplerGraphKey> & active_graph_key() const noexcept;
    const std::vector<FlowSamplerCacheState> & cache_states() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::modules
