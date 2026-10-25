#include "engine/framework/runtime/kv_cache.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace engine::runtime {

namespace {

void validate_cache_tensor(const core::TensorValue & tensor, const TransformerKVCacheOptions & options) {
    if (tensor.type == GGML_TYPE_F32) {
        return;
    }
    if (options.allow_f16_storage && tensor.type == GGML_TYPE_F16) {
        return;
    }
    if (options.allow_bf16_storage && tensor.type == GGML_TYPE_BF16) {
        return;
    }
    throw std::runtime_error(
        options.allow_f16_storage || options.allow_bf16_storage
            ? "TransformerKVCache supports only f32/f16/bf16 cache tensors when enabled"
            : "TransformerKVCache requires f32 cache tensors");
}

void write_cache_tensor(
    const core::TensorValue & tensor,
    const std::vector<float> & values,
    const TransformerKVCacheOptions & options) {
    validate_cache_tensor(tensor, options);
    if (tensor.type == GGML_TYPE_F32) {
        core::write_tensor_f32(tensor, values);
        return;
    }
    if (options.allow_f16_storage && tensor.type == GGML_TYPE_F16) {
        core::write_tensor_f16(tensor, values);
        return;
    }
    if (options.allow_bf16_storage && tensor.type == GGML_TYPE_BF16) {
        core::write_tensor_bf16(tensor, values);
        return;
    }
    throw std::runtime_error("TransformerKVCache requires f32 cache tensors");
}

void copy_cache_row(
    const std::vector<float> & source,
    int64_t source_row,
    std::vector<float> & dest,
    int64_t dest_row,
    int64_t step_elems) {
    const size_t src_begin = static_cast<size_t>(source_row * step_elems);
    const size_t dst_begin = static_cast<size_t>(dest_row * step_elems);
    const size_t count = static_cast<size_t>(step_elems);
    std::copy(source.begin() + static_cast<std::ptrdiff_t>(src_begin),
                source.begin() + static_cast<std::ptrdiff_t>(src_begin + count),
                dest.begin() + static_cast<std::ptrdiff_t>(dst_begin));
}

std::vector<float> read_cache_tensor(const core::TensorValue & tensor, const TransformerKVCacheOptions & options) {
    validate_cache_tensor(tensor, options);
    if (tensor.type == GGML_TYPE_F32) {
        return core::read_tensor_f32(tensor.tensor);
    }
    if (options.allow_f16_storage && tensor.type == GGML_TYPE_F16) {
        return core::read_tensor_f16(tensor.tensor);
    }
    if (options.allow_bf16_storage && tensor.type == GGML_TYPE_BF16) {
        return core::read_tensor_bf16(tensor.tensor);
    }
    throw std::runtime_error("TransformerKVCache requires f32 cache tensors");
}

}  // namespace

TransformerKVCache::TransformerKVCache(
    int64_t cache_steps,
    int64_t step_elems,
    std::vector<core::TensorValue> keys,
    std::vector<core::TensorValue> values)
    : TransformerKVCache(cache_steps, step_elems, std::move(keys), std::move(values), {}) {}

TransformerKVCache::TransformerKVCache(
    int64_t cache_steps,
    int64_t step_elems,
    std::vector<core::TensorValue> keys,
    std::vector<core::TensorValue> values,
    TransformerKVCacheOptions options)
    : cache_steps_(std::max<int64_t>(0, cache_steps)),
      step_elems_(std::max<int64_t>(0, step_elems)),
      options_(options) {
    if (step_elems_ <= 0) {
        throw std::runtime_error("TransformerKVCache requires positive step_elems");
    }
    if (options_.ring_mode) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("TransformerKVCache ring_mode requires positive cache_steps");
        }
        if (options_.ring_pinned_steps < 0 || options_.ring_pinned_steps >= cache_steps_) {
            throw std::runtime_error(
                "TransformerKVCache ring_pinned_steps must satisfy 0 <= pinned < cache_steps");
        }
    }
    if (keys.size() != values.size()) {
        throw std::runtime_error("TransformerKVCache key/value layer counts must match");
    }
    const size_t cache_elems = options_.lazy_import_scratch
        ? 0 : static_cast<size_t>(cache_steps_ * step_elems_);
    layers_.reserve(keys.size());
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        validate_cache_tensor(keys[layer], options_);
        validate_cache_tensor(values[layer], options_);
        layers_.push_back(LayerCache{
            std::move(keys[layer]),
            std::move(values[layer]),
            std::vector<float>(cache_elems, 0.0F),
            std::vector<float>(cache_elems, 0.0F),
        });
    }
}

void TransformerKVCache::clear_on_backend() {
    for (auto & layer : layers_) {
        ggml_backend_tensor_memset(layer.key_tensor.tensor, 0, 0, ggml_nbytes(layer.key_tensor.tensor));
        ggml_backend_tensor_memset(layer.value_tensor.tensor, 0, 0, ggml_nbytes(layer.value_tensor.tensor));
    }
    current_end_ = 0;
    valid_steps_ = 0;
}

void TransformerKVCache::import_state(const TransformerKVState & state) {
    current_end_ = state.current_end;
    if (layers_.empty()) {
        valid_steps_ = 0;
        return;
    }
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("TransformerKVCache state layer count does not match cache layer count");
    }
    const int64_t state_steps = state.layers.empty() ? 0 : state.layers.front().valid_steps;
    if (state_steps > cache_steps_) {
        throw std::runtime_error("TransformerKVCache state valid_steps exceeds cache capacity");
    }
    valid_steps_ = state_steps;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto & cache = layers_[layer];
        const auto & source = state.layers[layer];
        if (source.valid_steps != state_steps) {
            throw std::runtime_error("TransformerKVCache requires consistent valid_steps across all layers");
        }
        const size_t keep_elems = static_cast<size_t>(source.valid_steps * step_elems_);
        if (source.key.size() != source.value.size()) {
            throw std::runtime_error("TransformerKVCache source key/value sizes must match");
        }
        if (source.key.size() != keep_elems) {
            throw std::runtime_error("TransformerKVCache source tensors do not match valid_steps * step_elems");
        }
        if (cache_steps_ > 0) {
            const size_t cache_elems = static_cast<size_t>(cache_steps_ * step_elems_);
            cache.import_key_scratch.resize(cache_elems);
            cache.import_value_scratch.resize(cache_elems);
            std::fill(cache.import_key_scratch.begin(), cache.import_key_scratch.end(), 0.0F);
            std::fill(cache.import_value_scratch.begin(), cache.import_value_scratch.end(), 0.0F);
            if (keep_elems > 0) {
                if (!options_.ring_mode) {
                    std::copy(source.key.begin(), source.key.end(), cache.import_key_scratch.begin());
                    std::copy(source.value.begin(), source.value.end(), cache.import_value_scratch.begin());
                } else {
                    for (int64_t row = 0; row < state_steps; ++row) {
                        const int64_t position = ring_stored_position(
                            row, state_steps, current_end_, options_.ring_pinned_steps);
                        copy_cache_row(
                            source.key, row, cache.import_key_scratch, slot_for_position(position), step_elems_);
                        copy_cache_row(
                            source.value, row, cache.import_value_scratch, slot_for_position(position), step_elems_);
                    }
                }
            }
            write_cache_tensor(cache.key_tensor, cache.import_key_scratch, options_);
            write_cache_tensor(cache.value_tensor, cache.import_value_scratch, options_);
        }
    }
}

void TransformerKVCache::import_prefix(
    const std::vector<core::TensorValue> & keys,
    const std::vector<core::TensorValue> & values,
    int64_t steps) {
    if (steps <= 0 || steps > cache_steps_ || keys.size() != layers_.size() || values.size() != layers_.size()) {
        throw std::runtime_error("TransformerKVCache prefix shape exceeds cache or has different layers");
    }
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        for (const auto & tensors : {std::pair{keys[layer], layers_[layer].key_tensor},
                                    std::pair{values[layer], layers_[layer].value_tensor}}) {
            const auto & source = tensors.first;
            const auto & target = tensors.second;
            auto expected = target.shape;
            expected.dims[1] = steps;
            core::validate_shape(source, expected, "KV prefix");
            if (source.shape.dims[0] != 1 || source.type != target.type || !ggml_is_contiguous(source.tensor)) {
                throw std::runtime_error("TransformerKVCache prefix requires unbatched contiguous tensors of matching precision");
            }
            // A temporary descriptor limits the copy to the prefix without
            // adding per-request views to a retained ggml context.
            ggml_tensor destination = *target.tensor;
            std::copy(source.tensor->ne, source.tensor->ne + GGML_MAX_DIMS, destination.ne);
            std::copy(source.tensor->nb, source.tensor->nb + GGML_MAX_DIMS, destination.nb);
            ggml_backend_tensor_copy(source.tensor, &destination);
            const size_t copied = ggml_nbytes(source.tensor);
            const size_t tail = ggml_nbytes(target.tensor) - copied;
            if (tail != 0) ggml_backend_tensor_memset(target.tensor, 0, copied, tail);
        }
    }
    valid_steps_ = steps;
    current_end_ = steps;
}

TransformerKVState TransformerKVCache::export_state() const {
    TransformerKVState state;
    state.current_end = current_end_;
    state.layers.resize(layers_.size());
    const size_t keep_elems = static_cast<size_t>(valid_steps_ * step_elems_);
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto & out = state.layers[layer];
        out.valid_steps = valid_steps_;
        if (keep_elems == 0) {
            continue;
        }
        const auto key_values = read_cache_tensor(layers_[layer].key_tensor, options_);
        const auto value_values = read_cache_tensor(layers_[layer].value_tensor, options_);
        if (!options_.ring_mode) {
            out.key.assign(key_values.begin(), key_values.begin() + static_cast<ptrdiff_t>(keep_elems));
            out.value.assign(value_values.begin(), value_values.begin() + static_cast<ptrdiff_t>(keep_elems));
            continue;
        }
        out.key.resize(keep_elems);
        out.value.resize(keep_elems);
        for (int64_t row = 0; row < valid_steps_; ++row) {
            const int64_t position =
                ring_stored_position(row, valid_steps_, current_end_, options_.ring_pinned_steps);
            copy_cache_row(key_values, slot_for_position(position), out.key, row, step_elems_);
            copy_cache_row(value_values, slot_for_position(position), out.value, row, step_elems_);
        }
    }
    return state;
}

int64_t TransformerKVCache::slot_for_position(int64_t position) const {
    if (position < 0) {
        throw std::runtime_error("TransformerKVCache slot_for_position requires a non-negative position");
    }
    if (!options_.ring_mode || position < options_.ring_pinned_steps) {
        return position;
    }
    return options_.ring_pinned_steps +
        ((position - options_.ring_pinned_steps) % (cache_steps_ - options_.ring_pinned_steps));
}

void TransformerKVCache::advance_after_direct_append(int64_t steps) {
    if (steps <= 0) {
        return;
    }
    if (options_.ring_mode) {
        valid_steps_ = std::min(cache_steps_, valid_steps_ + steps);
        current_end_ += steps;
        return;
    }
    if (valid_steps_ + steps > cache_steps_) {
        throw std::runtime_error("TransformerKVCache direct append exceeds cache capacity");
    }
    valid_steps_ += steps;
    current_end_ += steps;
}

void TransformerKVCache::retain_prefix(int64_t prefix_steps) {
    if (prefix_steps < 0 || prefix_steps > valid_steps_) {
        throw std::runtime_error("TransformerKVCache prefix length exceeds current state");
    }
    valid_steps_ = prefix_steps;
    current_end_ = prefix_steps;
}

int64_t TransformerKVCache::valid_steps() const noexcept {
    return valid_steps_;
}

int64_t TransformerKVCache::current_end() const noexcept {
    return current_end_;
}

int64_t TransformerKVCache::cache_steps() const noexcept {
    return cache_steps_;
}

const core::TensorValue & TransformerKVCache::key_tensor(size_t layer) const {
    return layers_.at(layer).key_tensor;
}

const core::TensorValue & TransformerKVCache::value_tensor(size_t layer) const {
    return layers_.at(layer).value_tensor;
}

void TransformerKVCache::trace_log_state(const std::string & name, int64_t num_heads, int64_t head_dim) const {
    if (!debug::trace_log_enabled()) {
        return;
    }
    debug::trace_log_scalar(name + ".current_end", current_end_);
    if (layers_.empty() || valid_steps_ <= 0) {
        return;
    }
    const size_t keep_elems = static_cast<size_t>(valid_steps_ * step_elems_);
    const auto first_key = read_cache_tensor(layers_.front().key_tensor, options_);
    std::vector<float> first_key_keep(first_key.begin(), first_key.begin() + static_cast<ptrdiff_t>(keep_elems));
    debug::trace_log_f32(name + ".layer0.key", {1, valid_steps_, num_heads, head_dim}, first_key_keep);
    if (layers_.size() > 1) {
        const auto last_key = read_cache_tensor(layers_.back().key_tensor, options_);
        std::vector<float> last_key_keep(last_key.begin(), last_key.begin() + static_cast<ptrdiff_t>(keep_elems));
        debug::trace_log_f32(name + ".layer_last.key", {1, valid_steps_, num_heads, head_dim}, last_key_keep);
    }
}

TransformerBatchedKVCache::TransformerBatchedKVCache(
    int64_t cache_steps,
    int64_t batch_size,
    int64_t row_elems,
    std::vector<core::TensorValue> keys,
    std::vector<core::TensorValue> values)
    : TransformerBatchedKVCache(cache_steps, batch_size, row_elems, std::move(keys), std::move(values), {}) {}

TransformerBatchedKVCache::TransformerBatchedKVCache(
    int64_t cache_steps,
    int64_t batch_size,
    int64_t row_elems,
    std::vector<core::TensorValue> keys,
    std::vector<core::TensorValue> values,
    TransformerKVCacheOptions options)
    : cache_steps_(std::max<int64_t>(0, cache_steps)),
      batch_size_(std::max<int64_t>(0, batch_size)),
      row_elems_(std::max<int64_t>(0, row_elems)),
      options_(options) {
    if (cache_steps_ <= 0 || batch_size_ <= 0 || row_elems_ <= 0) {
        throw std::runtime_error("TransformerBatchedKVCache requires positive cache_steps, batch_size, and row_elems");
    }
    if (keys.size() != values.size()) {
        throw std::runtime_error("TransformerBatchedKVCache key/value layer counts must match");
    }
    const size_t cache_elems = static_cast<size_t>(batch_size_ * cache_steps_ * row_elems_);
    layers_.reserve(keys.size());
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        validate_cache_tensor(keys[layer], options_);
        validate_cache_tensor(values[layer], options_);
        layers_.push_back(LayerCache{
            std::move(keys[layer]),
            std::move(values[layer]),
            std::vector<float>(cache_elems, 0.0F),
            std::vector<float>(cache_elems, 0.0F),
        });
    }
}

void TransformerBatchedKVCache::import_state(const TransformerBatchedKVState & state) {
    if (state.batch_size != batch_size_) {
        throw std::runtime_error("TransformerBatchedKVCache state batch size does not match cache batch size");
    }
    if (!state.valid_steps_by_batch.empty() &&
        state.valid_steps_by_batch.size() != static_cast<size_t>(batch_size_)) {
        throw std::runtime_error("TransformerBatchedKVCache valid_steps_by_batch size mismatch");
    }
    if (!state.current_end_by_batch.empty() &&
        state.current_end_by_batch.size() != static_cast<size_t>(batch_size_)) {
        throw std::runtime_error("TransformerBatchedKVCache current_end_by_batch size mismatch");
    }
    current_end_by_batch_ = state.current_end_by_batch;
    valid_steps_by_batch_ = state.valid_steps_by_batch;
    current_end_ = current_end_by_batch_.empty()
        ? state.current_end
        : *std::max_element(current_end_by_batch_.begin(), current_end_by_batch_.end());
    if (layers_.empty()) {
        valid_steps_ = 0;
        return;
    }
    if (state.layers.size() != layers_.size()) {
        throw std::runtime_error("TransformerBatchedKVCache state layer count does not match cache layer count");
    }
    const int64_t state_steps = valid_steps_by_batch_.empty()
        ? (state.layers.empty() ? 0 : state.layers.front().valid_steps)
        : *std::max_element(valid_steps_by_batch_.begin(), valid_steps_by_batch_.end());
    if (state_steps > cache_steps_) {
        throw std::runtime_error("TransformerBatchedKVCache state valid_steps exceeds cache capacity");
    }
    valid_steps_ = state_steps;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto & cache = layers_[layer];
        const auto & source = state.layers[layer];
        if (valid_steps_by_batch_.empty() && source.valid_steps != state_steps) {
            throw std::runtime_error("TransformerBatchedKVCache requires consistent valid_steps across all layers");
        }
        std::fill(cache.import_key_scratch.begin(), cache.import_key_scratch.end(), 0.0F);
        std::fill(cache.import_value_scratch.begin(), cache.import_value_scratch.end(), 0.0F);
        const bool variable_rows = !valid_steps_by_batch_.empty();
        size_t src_offset = 0;
        if (!variable_rows) {
            const size_t source_row_elems = static_cast<size_t>(state_steps * row_elems_);
            const size_t state_elems = static_cast<size_t>(batch_size_) * source_row_elems;
            if (source.key.size() != source.value.size() || source.key.size() != state_elems) {
                throw std::runtime_error(
                    "TransformerBatchedKVCache source tensors do not match batch * valid_steps * row_elems");
            }
        }
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const int64_t row_steps = variable_rows ? valid_steps_by_batch_[static_cast<size_t>(batch)] : state_steps;
            if (row_steps < 0 || row_steps > state_steps) {
                throw std::runtime_error("TransformerBatchedKVCache row valid_steps is invalid");
            }
            const size_t copy_elems = static_cast<size_t>(row_steps * row_elems_);
            const size_t dst_offset = static_cast<size_t>(batch * cache_steps_ * row_elems_);
            if (!variable_rows) {
                src_offset = static_cast<size_t>(batch) * static_cast<size_t>(state_steps * row_elems_);
            }
            if (src_offset + copy_elems > source.key.size() || source.key.size() != source.value.size()) {
                throw std::runtime_error("TransformerBatchedKVCache compact source tensor size mismatch");
            }
            std::copy(
                source.key.begin() + static_cast<std::ptrdiff_t>(src_offset),
                source.key.begin() + static_cast<std::ptrdiff_t>(src_offset + copy_elems),
                cache.import_key_scratch.begin() + static_cast<std::ptrdiff_t>(dst_offset));
            std::copy(
                source.value.begin() + static_cast<std::ptrdiff_t>(src_offset),
                source.value.begin() + static_cast<std::ptrdiff_t>(src_offset + copy_elems),
                cache.import_value_scratch.begin() + static_cast<std::ptrdiff_t>(dst_offset));
            if (variable_rows) {
                src_offset += copy_elems;
            }
        }
        if (variable_rows && src_offset != source.key.size()) {
            throw std::runtime_error("TransformerBatchedKVCache compact source tensor has trailing values");
        }
        write_cache_tensor(cache.key_tensor, cache.import_key_scratch, options_);
        write_cache_tensor(cache.value_tensor, cache.import_value_scratch, options_);
    }
}

TransformerBatchedKVState TransformerBatchedKVCache::export_state() const {
    TransformerBatchedKVState state;
    state.batch_size = batch_size_;
    state.current_end = current_end_;
    state.current_end_by_batch = current_end_by_batch_;
    state.valid_steps_by_batch = valid_steps_by_batch_;
    state.layers.resize(layers_.size());
    const size_t copy_elems = static_cast<size_t>(valid_steps_ * row_elems_);
    const size_t state_elems = static_cast<size_t>(batch_size_) * copy_elems;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto & out = state.layers[layer];
        out.valid_steps = valid_steps_;
        out.key.resize(state_elems);
        out.value.resize(state_elems);
        if (copy_elems == 0) {
            continue;
        }
        const auto key_values = read_cache_tensor(layers_[layer].key_tensor, options_);
        const auto value_values = read_cache_tensor(layers_[layer].value_tensor, options_);
        for (int64_t batch = 0; batch < batch_size_; ++batch) {
            const size_t src_offset = static_cast<size_t>(batch * cache_steps_ * row_elems_);
            const size_t dst_offset = static_cast<size_t>(batch) * copy_elems;
            std::copy(
                key_values.begin() + static_cast<std::ptrdiff_t>(src_offset),
                key_values.begin() + static_cast<std::ptrdiff_t>(src_offset + copy_elems),
                out.key.begin() + static_cast<std::ptrdiff_t>(dst_offset));
            std::copy(
                value_values.begin() + static_cast<std::ptrdiff_t>(src_offset),
                value_values.begin() + static_cast<std::ptrdiff_t>(src_offset + copy_elems),
                out.value.begin() + static_cast<std::ptrdiff_t>(dst_offset));
        }
    }
    return state;
}

void TransformerBatchedKVCache::advance_after_direct_append(int64_t steps) {
    if (steps <= 0) {
        return;
    }
    if (valid_steps_ + steps > cache_steps_) {
        throw std::runtime_error("TransformerBatchedKVCache direct append exceeds cache capacity");
    }
    valid_steps_ += steps;
    current_end_ += steps;
    for (auto & value : valid_steps_by_batch_) {
        value += steps;
    }
    for (auto & value : current_end_by_batch_) {
        value += steps;
    }
}

int64_t TransformerBatchedKVCache::batch_size() const noexcept {
    return batch_size_;
}

int64_t TransformerBatchedKVCache::valid_steps() const noexcept {
    return valid_steps_;
}

int64_t TransformerBatchedKVCache::current_end() const noexcept {
    return current_end_;
}

int64_t TransformerBatchedKVCache::cache_steps() const noexcept {
    return cache_steps_;
}

const std::vector<int64_t> & TransformerBatchedKVCache::valid_steps_by_batch() const noexcept {
    return valid_steps_by_batch_;
}

const std::vector<int64_t> & TransformerBatchedKVCache::current_end_by_batch() const noexcept {
    return current_end_by_batch_;
}

core::TensorValue view_transformer_kv_cache_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim,
    const char * label,
    ggml_type view_type) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error(std::string(label) + " cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            head_dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, head_dim}),
        view_type);
}

}  // namespace engine::runtime
