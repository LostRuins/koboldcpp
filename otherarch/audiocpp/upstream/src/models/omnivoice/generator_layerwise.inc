// This file is included by generator.cpp inside the OmniVoice anonymous namespace.

class GraphBufferSet {
public:
    ~GraphBufferSet() {
        clear();
    }

    void clear() noexcept {
        for (auto & buffer : buffers_) {
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
        }
    }

    void allocate(ggml_context * ctx, ggml_backend_t backend, const char * error_prefix) {
        if (ctx == nullptr || backend == nullptr) {
            throw std::runtime_error(std::string(error_prefix) + " requires initialized graph context and backend");
        }
        auto * const buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const size_t max_size = ggml_backend_buft_get_max_size(buft);
        if (alignment == 0 || max_size == 0) {
            throw std::runtime_error(std::string(error_prefix) + " allocator returned invalid backend limits");
        }

        struct TensorRangePlan {
            ggml_tensor * first = nullptr;
            ggml_tensor * last = nullptr;
            size_t bytes = 0;
        };

        std::vector<TensorRangePlan> plans;
        plans.reserve(8);

        size_t current_bytes = 0;
        ggml_tensor * range_first = ggml_get_first_tensor(ctx);
        for (ggml_tensor * tensor = range_first; tensor != nullptr; tensor = ggml_get_next_tensor(ctx, tensor)) {
            size_t tensor_bytes = 0;
            if (tensor->data == nullptr && tensor->view_src == nullptr) {
                tensor_bytes = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, tensor), alignment);
            }

            if (current_bytes > 0 && (current_bytes + tensor_bytes) > max_size) {
                plans.push_back({range_first, tensor, current_bytes});
                range_first = tensor;
                current_bytes = tensor_bytes;
            } else {
                current_bytes += tensor_bytes;
            }
        }
        if (current_bytes > 0) {
            plans.push_back({range_first, nullptr, current_bytes});
        }
        if (plans.empty()) {
            throw std::runtime_error(std::string(error_prefix) + " requires non-zero compute buffer");
        }

        if (buffers_.size() < plans.size()) {
            buffers_.resize(plans.size(), nullptr);
        }

        for (size_t i = 0; i < plans.size(); ++i) {
            auto & buffer = buffers_[i];
            if (buffer == nullptr || ggml_backend_buffer_get_size(buffer) < plans[i].bytes) {
                if (buffer != nullptr) {
                    ggml_backend_buffer_free(buffer);
                    buffer = nullptr;
                }
                buffer = ggml_backend_buft_alloc_buffer(buft, plans[i].bytes);
                if (buffer == nullptr) {
                    throw std::runtime_error(std::string("failed to allocate ") + error_prefix + " graph buffer");
                }
                ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            } else {
                ggml_backend_buffer_reset(buffer);
            }

            auto tallocr = ggml_tallocr_new(buffer);
            for (ggml_tensor * tensor = plans[i].first; tensor != plans[i].last; tensor = ggml_get_next_tensor(ctx, tensor)) {
                enum ggml_status status = GGML_STATUS_SUCCESS;
                if (tensor->data == nullptr) {
                    if (tensor->view_src == nullptr) {
                        status = ggml_tallocr_alloc(&tallocr, tensor);
                    } else if (tensor->buffer == nullptr) {
                        status = ggml_backend_view_init(tensor);
                    }
                } else if (tensor->view_src != nullptr && tensor->buffer == nullptr) {
                    status = ggml_backend_view_init(tensor);
                }
                if (status != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error(std::string("failed to allocate ") + error_prefix + " graph tensor");
                }
            }
        }
    }

private:
    std::vector<ggml_backend_buffer_t> buffers_;
};

class LayerwiseEmbeddingStage {
public:
    ~LayerwiseEmbeddingStage() {
        clear_graph();
    }

    void rebuild(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t total_token_capacity) {
        clear_graph();
        runtime_ = std::move(runtime);
        graph_arena_bytes_ = graph_arena_bytes;
        total_tokens_capacity_ = total_token_capacity;
        if (runtime_ == nullptr || total_tokens_capacity_ <= 0) {
            throw std::runtime_error("OmniVoice layerwise embedding stage requires valid runtime and token capacity");
        }

        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice layerwise embedding graph context");
        }

        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.layerwise.embedding", runtime_->backend_type()};
        std::array<core::TensorValue, 8> audio_id_values = {};
        auto text_ids =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2, total_tokens_capacity_}));
        text_ids_ = text_ids.tensor;
        ggml_set_input(text_ids_);
        for (int64_t codebook = 0; codebook < config.num_audio_codebook; ++codebook) {
            auto audio_ids =
                core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({2, total_tokens_capacity_}));
            audio_ids_[static_cast<size_t>(codebook)] = audio_ids.tensor;
            ggml_set_input(audio_ids_[static_cast<size_t>(codebook)]);
            audio_id_values[static_cast<size_t>(codebook)] = audio_ids;
        }
        auto audio_mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, total_tokens_capacity_, 1}));
        auto text_mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, total_tokens_capacity_, 1}));
        audio_mask_ = audio_mask.tensor;
        text_mask_ = text_mask.tensor;
        ggml_set_input(audio_mask_);
        ggml_set_input(text_mask_);
        output_ = ensure_contiguous(ctx, build_embeddings(ctx, config, weights, text_ids, audio_id_values, audio_mask, text_mask)).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        buffers_.allocate(ctx_.get(), runtime_->backend(), "OmniVoice layerwise embedding");
    }

    void run(
        const std::vector<int32_t> & text_ids,
        const std::array<std::vector<int32_t>, 8> & audio_ids,
        const std::vector<float> & audio_mask,
        const std::vector<float> & text_mask,
        std::vector<float> & output,
        double & compute_ms,
        double & readback_ms) {
        if (graph_ == nullptr || output_ == nullptr || runtime_ == nullptr) {
            throw std::runtime_error("OmniVoice layerwise embedding stage is not built");
        }
        ggml_backend_tensor_set(text_ids_, text_ids.data(), 0, text_ids.size() * sizeof(int32_t));
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & ids = audio_ids[static_cast<size_t>(codebook)];
            ggml_backend_tensor_set(
                audio_ids_[static_cast<size_t>(codebook)],
                ids.data(),
                0,
                ids.size() * sizeof(int32_t));
        }
        ggml_backend_tensor_set(audio_mask_, audio_mask.data(), 0, audio_mask.size() * sizeof(float));
        ggml_backend_tensor_set(text_mask_, text_mask.data(), 0, text_mask.size() * sizeof(float));
        compute_graph(compute_ms);
        const size_t hidden_count = static_cast<size_t>(
            2 * total_tokens_capacity_ * runtime_->assets().config.llm.hidden_size);
        if (output.size() != hidden_count) {
            output.assign(hidden_count, 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
    }

    void clear_graph() {
        if (runtime_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        }
        graph_ = nullptr;
        output_ = nullptr;
        audio_mask_ = nullptr;
        text_mask_ = nullptr;
        for (auto & tensor : audio_ids_) {
            tensor = nullptr;
        }
        text_ids_ = nullptr;
        ctx_.reset();
    }

private:
    void compute_graph(double & compute_ms) {
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice layerwise embedding graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);
    }

    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t total_tokens_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * text_ids_ = nullptr;
    std::array<ggml_tensor *, 8> audio_ids_{};
    ggml_tensor * audio_mask_ = nullptr;
    ggml_tensor * text_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    GraphBufferSet buffers_;
};

class LayerwiseDecoderLayerStage {
public:
    ~LayerwiseDecoderLayerStage() {
        clear_graph();
    }

    void rebuild(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t total_token_capacity,
        int64_t layer_index) {
        clear_graph();
        runtime_ = std::move(runtime);
        graph_arena_bytes_ = graph_arena_bytes;
        total_tokens_capacity_ = total_token_capacity;
        layer_index_ = layer_index;
        if (runtime_ == nullptr || total_tokens_capacity_ <= 0) {
            throw std::runtime_error("OmniVoice layerwise decoder stage requires valid runtime and token capacity");
        }
        const auto & config = runtime_->assets().config;
        if (layer_index_ < 0 || layer_index_ >= static_cast<int64_t>(runtime_->weights().layers.size())) {
            throw std::runtime_error("OmniVoice layerwise decoder layer index is out of range");
        }

        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice layerwise decoder graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.layerwise.decoder", runtime_->backend_type()};
        auto input =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, total_tokens_capacity_, config.llm.hidden_size}));
        input_ = input.tensor;
        ggml_set_input(input_);
        positions_ =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({total_tokens_capacity_})).tensor;
        ggml_set_input(positions_);
        auto positions =
            core::wrap_tensor(positions_, core::TensorShape::from_dims({total_tokens_capacity_}), GGML_TYPE_I32);
        auto attention_mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, 1, total_tokens_capacity_, total_tokens_capacity_}));
        attention_mask_ = attention_mask.tensor;
        ggml_set_input(attention_mask_);
        auto output = decoder_layer(
            ctx,
            input,
            positions,
            runtime_->weights().layers[static_cast<size_t>(layer_index_)],
            config.llm,
            attention_mask);
        output_ = ensure_contiguous(ctx, output).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        buffers_.allocate(ctx_.get(), runtime_->backend(), "OmniVoice layerwise decoder");
    }

    void run(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t total_token_capacity,
        int64_t layer_index,
        const std::vector<int32_t> & positions,
        const std::vector<float> & attention_mask,
        const std::vector<float> & input,
        std::vector<float> & output,
        double & compute_ms,
        double & readback_ms) {
        rebuild(std::move(runtime), graph_arena_bytes, total_token_capacity, layer_index);
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(attention_mask_, attention_mask.data(), 0, attention_mask.size() * sizeof(float));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice layerwise decoder graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);
        if (output.size() != input.size()) {
            output.assign(input.size(), 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
    }

    void clear_graph() {
        if (runtime_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        }
        graph_ = nullptr;
        output_ = nullptr;
        attention_mask_ = nullptr;
        positions_ = nullptr;
        input_ = nullptr;
        ctx_.reset();
    }

private:
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t layer_index_ = -1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    GraphBufferSet buffers_;
};

class LayerwiseHeadStage {
public:
    ~LayerwiseHeadStage() {
        clear_graph();
    }

    void rebuild(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t total_token_capacity,
        int64_t target_frame_capacity) {
        clear_graph();
        runtime_ = std::move(runtime);
        graph_arena_bytes_ = graph_arena_bytes;
        total_tokens_capacity_ = total_token_capacity;
        target_frame_capacity_ = target_frame_capacity;
        if (runtime_ == nullptr || total_tokens_capacity_ <= 0 || target_frame_capacity_ <= 0) {
            throw std::runtime_error("OmniVoice layerwise head stage requires valid runtime and capacities");
        }

        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize OmniVoice layerwise head graph context");
        }

        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "omnivoice.generator.layerwise.head", runtime_->backend_type()};
        auto input =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, total_tokens_capacity_, config.llm.hidden_size}));
        input_ = input.tensor;
        ggml_set_input(input_);
        conditional_target_indices_ =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({target_frame_capacity_})).tensor;
        unconditional_target_indices_ =
            core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({target_frame_capacity_})).tensor;
        guidance_scale_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1})).tensor;
        ggml_set_input(conditional_target_indices_);
        ggml_set_input(unconditional_target_indices_);
        ggml_set_input(guidance_scale_);

        auto x = modules::RMSNormModule({config.llm.hidden_size, config.llm.rms_norm_eps, true, false})
                     .build(ctx, input, binding::norm_data(ctx, weights.norm));
        auto conditional_hidden = modules::SliceModule({0, 0, 1}).build(ctx, x);
        conditional_hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, conditional_hidden),
            core::TensorShape::from_dims({total_tokens_capacity_, config.llm.hidden_size}));
        auto conditional_index_value = core::wrap_tensor(
            conditional_target_indices_,
            core::TensorShape::from_dims({target_frame_capacity_}),
            GGML_TYPE_I32);
        conditional_hidden = modules::EmbeddingModule({total_tokens_capacity_, config.llm.hidden_size})
                                 .build(ctx, conditional_index_value, conditional_hidden);
        conditional_hidden = core::reshape_tensor(
            ctx,
            conditional_hidden,
            core::TensorShape::from_dims({1, target_frame_capacity_, config.llm.hidden_size}));
        auto unconditional_hidden = modules::SliceModule({0, 1, 1}).build(ctx, x);
        unconditional_hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, unconditional_hidden),
            core::TensorShape::from_dims({total_tokens_capacity_, config.llm.hidden_size}));
        auto unconditional_index_value = core::wrap_tensor(
            unconditional_target_indices_,
            core::TensorShape::from_dims({target_frame_capacity_}),
            GGML_TYPE_I32);
        unconditional_hidden = modules::EmbeddingModule({total_tokens_capacity_, config.llm.hidden_size})
                                   .build(ctx, unconditional_index_value, unconditional_hidden);
        unconditional_hidden = core::reshape_tensor(
            ctx,
            unconditional_hidden,
            core::TensorShape::from_dims({1, target_frame_capacity_, config.llm.hidden_size}));
        auto hidden = modules::ConcatModule({0}).build(ctx, conditional_hidden, unconditional_hidden);
        auto logits = modules::LinearModule(
                          binding::linear_config(
                              config.llm.hidden_size,
                              config.num_audio_codebook * config.audio_vocab_size,
                              false))
                          .build(ctx, hidden, binding::linear_data(ctx, weights.audio_head));
        auto conditional_logits = modules::SliceModule({0, 0, 1}).build(ctx, logits);
        auto unconditional_logits = modules::SliceModule({0, 1, 1}).build(ctx, logits);
        auto guidance_scale_value = core::wrap_tensor(
            guidance_scale_,
            core::TensorShape::from_dims({1}),
            GGML_TYPE_F32);
        guidance_scale_value = core::reshape_tensor(ctx, guidance_scale_value, core::TensorShape::from_dims({1, 1, 1}));
        auto guidance_scale_expanded = modules::RepeatModule({conditional_logits.shape}).build(ctx, guidance_scale_value);
        auto diff = core::wrap_tensor(
            ggml_sub(ctx.ggml, conditional_logits.tensor, unconditional_logits.tensor),
            conditional_logits.shape,
            GGML_TYPE_F32);
        auto scaled_diff = modules::MulModule{}.build(ctx, diff, guidance_scale_expanded);
        auto combined_raw = modules::AddModule{}.build(ctx, conditional_logits, scaled_diff);
        logits_ = ensure_contiguous(ctx, combined_raw).tensor;
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffers_.allocate(ctx_.get(), runtime_->backend(), "OmniVoice layerwise head");
    }

    void run(
        const std::vector<float> & hidden,
        const std::vector<int32_t> & conditional_indices,
        const std::vector<int32_t> & unconditional_indices,
        float guidance_scale,
        std::vector<float> & logits,
        double & compute_ms,
        double & readback_ms) {
        if (graph_ == nullptr || logits_ == nullptr || runtime_ == nullptr) {
            throw std::runtime_error("OmniVoice layerwise head stage is not built");
        }
        ggml_backend_tensor_set(input_, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(conditional_target_indices_, conditional_indices.data(), 0, conditional_indices.size() * sizeof(int32_t));
        ggml_backend_tensor_set(unconditional_target_indices_, unconditional_indices.data(), 0, unconditional_indices.size() * sizeof(int32_t));
        ggml_backend_tensor_set(guidance_scale_, &guidance_scale, 0, sizeof(guidance_scale));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OmniVoice layerwise head graph compute failed");
        }
        compute_ms += engine::debug::elapsed_ms(compute_start, compute_end);
        const size_t logits_count = static_cast<size_t>(
            target_frame_capacity_ * runtime_->assets().config.num_audio_codebook * runtime_->assets().config.audio_vocab_size);
        if (logits.size() != logits_count) {
            logits.assign(logits_count, 0.0F);
        }
        const auto readback_start = Clock::now();
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        const auto readback_end = Clock::now();
        readback_ms += engine::debug::elapsed_ms(readback_start, readback_end);
    }

    void clear_graph() {
        if (runtime_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        }
        graph_ = nullptr;
        logits_ = nullptr;
        guidance_scale_ = nullptr;
        unconditional_target_indices_ = nullptr;
        conditional_target_indices_ = nullptr;
        input_ = nullptr;
        ctx_.reset();
    }

private:
    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t target_frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * conditional_target_indices_ = nullptr;
    ggml_tensor * unconditional_target_indices_ = nullptr;
    ggml_tensor * guidance_scale_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    GraphBufferSet buffers_;
};

class Qwen3DiffusionLayerwiseGraph final : public Qwen3DiffusionGraph {
public:
    Qwen3DiffusionLayerwiseGraph(
        std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime,
        size_t graph_arena_bytes,
        int64_t total_token_capacity,
        int64_t target_frame_capacity)
        : runtime_(std::move(runtime)),
          graph_arena_bytes_(graph_arena_bytes) {
        rebuild(total_token_capacity, target_frame_capacity);
    }

    bool matches(const Qwen3DiffusionWeightsRuntime & runtime, int64_t total_tokens, int64_t target_frames) const override {
        return runtime_.get() == &runtime &&
            total_tokens_capacity_ == total_tokens &&
            target_frame_capacity_ == target_frames;
    }

    void rebuild(int64_t total_token_capacity, int64_t target_frame_capacity) override {
        if (total_token_capacity <= 0 || target_frame_capacity <= 0) {
            throw std::runtime_error("OmniVoice layerwise generator capacities are invalid");
        }
        const auto clear_start = Clock::now();
        embedding_.clear_graph();
        decoder_.clear_graph();
        head_.clear_graph();
        const auto clear_end = Clock::now();
        rebuild_clear_ms_ = engine::debug::elapsed_ms(clear_start, clear_end);

        total_tokens_capacity_ = total_token_capacity;
        target_frame_capacity_ = target_frame_capacity;

        const auto build_start = Clock::now();
        embedding_.rebuild(runtime_, graph_arena_bytes_, total_tokens_capacity_);
        head_.rebuild(runtime_, graph_arena_bytes_, total_tokens_capacity_, target_frame_capacity_);
        const auto build_end = Clock::now();
        rebuild_build_ms_ = engine::debug::elapsed_ms(build_start, build_end);
        rebuild_alloc_ms_ = 0.0;

        const auto init_start = Clock::now();
        position_host_.assign(static_cast<size_t>(total_tokens_capacity_), 0);
        for (int64_t i = 0; i < total_tokens_capacity_; ++i) {
            position_host_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        text_ids_host_.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0);
        conditional_target_indices_host_.assign(static_cast<size_t>(target_frame_capacity_), 0);
        unconditional_target_indices_host_.assign(static_cast<size_t>(target_frame_capacity_), 0);
        audio_mask_values_host_.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0.0F);
        text_mask_values_host_.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0.0F);
        attention_values_host_.assign(
            static_cast<size_t>(2 * total_tokens_capacity_ * total_tokens_capacity_),
            kMaskedAttentionBias);
        for (auto & ids : audio_ids_host_) {
            ids.assign(static_cast<size_t>(2 * total_tokens_capacity_), 0);
        }
        hidden_a_.clear();
        hidden_b_.clear();
        logits_host_.clear();
        current_total_tokens_ = 0;
        current_target_frames_ = 0;
        current_conditional_target_start_ = 0;
        last_target_frames_ = -1;
        last_conditional_target_start_ = -1;
        last_mask_conditional_total_ = -1;
        last_mask_conditional_audio_start_ = -1;
        last_mask_unconditional_total_ = -1;
        guidance_scale_host_ = 0.0F;
        rebuild_init_ms_ = engine::debug::elapsed_ms(init_start, Clock::now());
    }

    void prepare_request(const PackedInputs & inputs) override {
        const int64_t total_tokens =
            inputs.style_tokens + inputs.text_tokens + inputs.reference_frames + inputs.target_frames;
        if (total_tokens <= 0 ||
            total_tokens > total_tokens_capacity_ ||
            inputs.target_frames > target_frame_capacity_) {
            throw std::runtime_error("OmniVoice layerwise packed input shape exceeds graph capacity");
        }
        current_total_tokens_ = total_tokens;
        current_target_frames_ = inputs.target_frames;
        current_conditional_target_start_ =
            inputs.style_tokens + inputs.text_tokens + inputs.reference_frames;
        std::fill(
            text_ids_host_.begin(),
            text_ids_host_.end(),
            static_cast<int32_t>(runtime_->assets().config.audio_mask_id));
        for (int64_t pos = 0; pos < total_tokens; ++pos) {
            text_ids_host_[static_cast<size_t>(pos)] = inputs.conditional_text_ids[static_cast<size_t>(pos)];
            text_ids_host_[static_cast<size_t>(total_tokens_capacity_ + pos)] =
                inputs.unconditional_text_ids[static_cast<size_t>(pos)];
        }
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & conditional_ids = inputs.conditional_audio_ids[static_cast<size_t>(codebook)];
            const auto & unconditional_ids = inputs.unconditional_audio_ids[static_cast<size_t>(codebook)];
            auto & ids = audio_ids_host_[static_cast<size_t>(codebook)];
            std::fill(
                ids.begin(),
                ids.end(),
                static_cast<int32_t>(codebook * runtime_->assets().config.audio_vocab_size +
                                     runtime_->assets().config.audio_mask_id));
            for (int64_t pos = 0; pos < total_tokens; ++pos) {
                ids[static_cast<size_t>(pos)] = conditional_ids[static_cast<size_t>(pos)];
                ids[static_cast<size_t>(total_tokens_capacity_ + pos)] =
                    unconditional_ids[static_cast<size_t>(pos)];
            }
        }
        if (last_conditional_target_start_ != current_conditional_target_start_ ||
            last_target_frames_ != current_target_frames_) {
            for (int64_t frame = 0; frame < inputs.target_frames; ++frame) {
                conditional_target_indices_host_[static_cast<size_t>(frame)] =
                    static_cast<int32_t>(current_conditional_target_start_ + frame);
                unconditional_target_indices_host_[static_cast<size_t>(frame)] =
                    static_cast<int32_t>(frame);
            }
            last_conditional_target_start_ = current_conditional_target_start_;
            last_target_frames_ = current_target_frames_;
        }
        const int64_t conditional_total =
            inputs.style_tokens + inputs.text_tokens + inputs.reference_frames + inputs.target_frames;
        const int64_t conditional_audio_start = inputs.style_tokens + inputs.text_tokens;
        const int64_t unconditional_total = inputs.target_frames;
        if (last_mask_conditional_total_ != conditional_total ||
            last_mask_conditional_audio_start_ != conditional_audio_start ||
            last_mask_unconditional_total_ != unconditional_total) {
            rebuild_runtime_masks(conditional_total, conditional_audio_start, unconditional_total);
            last_mask_conditional_total_ = conditional_total;
            last_mask_conditional_audio_start_ = conditional_audio_start;
            last_mask_unconditional_total_ = unconditional_total;
        }
    }

    void set_guidance_scale(float value) noexcept override {
        guidance_scale_host_ = value;
    }

    void update_generated_target_tokens(const PackedInputs & inputs) override {
        if (current_total_tokens_ <= 0 || current_target_frames_ != inputs.target_frames) {
            throw std::runtime_error("OmniVoice layerwise target token update requires a prepared request");
        }
        for (int64_t codebook = 0; codebook < runtime_->assets().config.num_audio_codebook; ++codebook) {
            const auto & conditional_ids = inputs.conditional_audio_ids[static_cast<size_t>(codebook)];
            const auto & unconditional_ids = inputs.unconditional_audio_ids[static_cast<size_t>(codebook)];
            auto & host_ids = audio_ids_host_[static_cast<size_t>(codebook)];
            for (int64_t frame = 0; frame < current_target_frames_; ++frame) {
                host_ids[static_cast<size_t>(current_conditional_target_start_ + frame)] =
                    conditional_ids[static_cast<size_t>(current_conditional_target_start_ + frame)];
                host_ids[static_cast<size_t>(total_tokens_capacity_ + frame)] =
                    unconditional_ids[static_cast<size_t>(frame)];
            }
        }
    }

    const std::vector<float> & compute_logits(double & compute_ms, double & readback_ms) override {
        embedding_.run(
            text_ids_host_,
            audio_ids_host_,
            audio_mask_values_host_,
            text_mask_values_host_,
            hidden_a_,
            compute_ms,
            readback_ms);
        auto * input = &hidden_a_;
        auto * output = &hidden_b_;
        const int64_t layers = static_cast<int64_t>(runtime_->weights().layers.size());
        for (int64_t layer = 0; layer < layers; ++layer) {
            decoder_.run(
                runtime_,
                graph_arena_bytes_,
                total_tokens_capacity_,
                layer,
                position_host_,
                attention_values_host_,
                *input,
                *output,
                compute_ms,
                readback_ms);
            std::swap(input, output);
        }
        head_.run(
            *input,
            conditional_target_indices_host_,
            unconditional_target_indices_host_,
            guidance_scale_host_,
            logits_host_,
            compute_ms,
            readback_ms);
        return logits_host_;
    }

    int64_t total_token_capacity() const noexcept override { return total_tokens_capacity_; }
    int64_t target_frame_capacity() const noexcept override { return target_frame_capacity_; }
    double rebuild_clear_ms() const noexcept override { return rebuild_clear_ms_; }
    double rebuild_build_ms() const noexcept override { return rebuild_build_ms_; }
    double rebuild_alloc_ms() const noexcept override { return rebuild_alloc_ms_; }
    double rebuild_init_ms() const noexcept override { return rebuild_init_ms_; }

private:
    void rebuild_runtime_masks(
        int64_t conditional_total,
        int64_t conditional_audio_start,
        int64_t unconditional_total) {
        std::fill(audio_mask_values_host_.begin(), audio_mask_values_host_.end(), 0.0F);
        std::fill(text_mask_values_host_.begin(), text_mask_values_host_.end(), 1.0F);
        for (int64_t pos = 0; pos < conditional_total; ++pos) {
            if (pos >= conditional_audio_start) {
                audio_mask_values_host_[static_cast<size_t>(pos)] = 1.0F;
                text_mask_values_host_[static_cast<size_t>(pos)] = 0.0F;
            }
        }
        for (int64_t pos = 0; pos < unconditional_total; ++pos) {
            const size_t offset = static_cast<size_t>(total_tokens_capacity_ + pos);
            audio_mask_values_host_[offset] = 1.0F;
            text_mask_values_host_[offset] = 0.0F;
        }
        std::fill(attention_values_host_.begin(), attention_values_host_.end(), kMaskedAttentionBias);
        const auto write_zero = [&](int batch, int64_t q, int64_t k) {
            const size_t index = static_cast<size_t>(
                k + total_tokens_capacity_ * (q + total_tokens_capacity_ * batch));
            attention_values_host_[index] = 0.0F;
        };
        for (int64_t q = 0; q < conditional_total; ++q) {
            for (int64_t k = 0; k < conditional_total; ++k) {
                write_zero(0, q, k);
            }
        }
        for (int64_t q = conditional_total; q < total_tokens_capacity_; ++q) {
            write_zero(0, q, q);
        }
        for (int64_t q = 0; q < unconditional_total; ++q) {
            for (int64_t k = 0; k < unconditional_total; ++k) {
                write_zero(1, q, k);
            }
        }
        for (int64_t q = unconditional_total; q < total_tokens_capacity_; ++q) {
            write_zero(1, q, q);
        }
    }

    std::shared_ptr<Qwen3DiffusionWeightsRuntime> runtime_;
    size_t graph_arena_bytes_ = 0;
    int64_t total_tokens_capacity_ = 0;
    int64_t target_frame_capacity_ = 0;
    int64_t current_total_tokens_ = 0;
    int64_t current_target_frames_ = 0;
    int64_t current_conditional_target_start_ = 0;
    int64_t last_target_frames_ = -1;
    int64_t last_conditional_target_start_ = -1;
    int64_t last_mask_conditional_total_ = -1;
    int64_t last_mask_conditional_audio_start_ = -1;
    int64_t last_mask_unconditional_total_ = -1;
    std::vector<int32_t> position_host_;
    std::vector<int32_t> text_ids_host_;
    std::array<std::vector<int32_t>, 8> audio_ids_host_{};
    std::vector<int32_t> conditional_target_indices_host_;
    std::vector<int32_t> unconditional_target_indices_host_;
    std::vector<float> audio_mask_values_host_;
    std::vector<float> text_mask_values_host_;
    std::vector<float> attention_values_host_;
    std::vector<float> hidden_a_;
    std::vector<float> hidden_b_;
    std::vector<float> logits_host_;
    float guidance_scale_host_ = 0.0F;
    double rebuild_clear_ms_ = 0.0;
    double rebuild_build_ms_ = 0.0;
    double rebuild_alloc_ms_ = 0.0;
    double rebuild_init_ms_ = 0.0;
    LayerwiseEmbeddingStage embedding_;
    LayerwiseDecoderLayerStage decoder_;
    LayerwiseHeadStage head_;
};
