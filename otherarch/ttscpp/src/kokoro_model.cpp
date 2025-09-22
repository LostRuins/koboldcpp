#include "kokoro_model.h"
#include <regex>

static struct ggml_tensor * build_albert_attn_mask(ggml_context * ctx, struct kokoro_duration_context *kctx, const kokoro_ubatch & batch) {
    kctx->attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) batch.n_tokens, (int64_t) batch.n_tokens);
    ggml_set_input(kctx->attn_mask);

    return kctx->attn_mask;
}

static struct ggml_tensor * build_albert_inputs(ggml_context * ctx, kokoro_model * model, ggml_tensor * input_tokens, ggml_tensor * positions, ggml_tensor * token_types) {
	struct ggml_tensor * tinpts = ggml_cont(ctx, ggml_get_rows(ctx, model->token_embd, input_tokens));
	struct ggml_tensor * pinpts = ggml_get_rows(ctx, model->position_embd, positions);

	struct ggml_tensor * inpts = ggml_cont(ctx, ggml_add(ctx, tinpts, pinpts));
	if (!model->static_token_types) {
		// Token type embeddings are actually static for kokoro at the moment, so we should never need to compute this on the fly.
		return ggml_add(ctx, inpts, ggml_get_rows(ctx, model->token_type_embd, token_types));
	}
	struct ggml_tensor * ainpts = ggml_add(ctx, inpts, model->static_token_type_values);

	struct ggml_tensor * out = ggml_cont(ctx, build_albert_norm(ctx, ainpts, model->input_norm_weight, model->input_norm_bias));
	return ggml_add(ctx, ggml_mul_mat(ctx, model->embd_hidden, out), model->embd_hidden_bias);
}

static struct ggml_tensor * build_albert_norm(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * bias) {
	// this is the standard eps for Albert
	float eps = 0.000000000001;
    cur = ggml_norm(ctx, cur, eps);
    cur = ggml_cont(ctx, ggml_add(ctx, ggml_mul(ctx, cur, weight), bias));
    return cur;
}

static struct ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, std::vector<ggml_tensor*> weights, std::vector<ggml_tensor*> biases, uint32_t sequence_length, bool reversed = false);

static struct ggml_tensor * build_lstm(ggml_context * ctx, ggml_tensor * input, lstm* rnn, uint32_t sequence_length, ggml_cgraph * gf) {
	struct ggml_tensor * resp = input;
	struct ggml_tensor * reverse_resp = input;

	// iterate over cells first so that at each pass to the next cell we have a fully formed vector (this improves performance as well as allocation for stacked lstms)
	for (int c = 0; c < rnn->cells.size(); c++) {
		ggml_build_forward_expand(gf, resp);
		resp = build_lstm_run(ctx, gf, resp, rnn->hidden[c], rnn->states[c], rnn->cells[c]->weights, rnn->cells[c]->biases, sequence_length);
		if (rnn->bidirectional) {
			reverse_resp = build_lstm_run(ctx, gf, reverse_resp, rnn->hidden[c], rnn->states[c], rnn->cells[c]->reverse_weights, rnn->cells[c]->reverse_biases, sequence_length, true);
		}
	}
	if (rnn->bidirectional) {
		resp = ggml_concat(ctx, resp, reverse_resp, 0);
	}
	return resp;
}

static struct ggml_tensor * build_lstm_run(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * input, ggml_tensor * h_0, ggml_tensor * c_0, std::vector<ggml_tensor*> weights, std::vector<ggml_tensor*> biases, uint32_t sequence_length, bool reversed) {
	struct ggml_tensor * I = ggml_add(ctx, ggml_mul_mat(ctx, weights[0], input), biases[0]);
	struct ggml_tensor * F = ggml_add(ctx, ggml_mul_mat(ctx, weights[2], input), biases[2]);
	struct ggml_tensor * G = ggml_add(ctx, ggml_mul_mat(ctx, weights[4], input), biases[4]);
	struct ggml_tensor * O = ggml_add(ctx, ggml_mul_mat(ctx, weights[6], input), biases[6]);

	struct ggml_tensor * outputs;

	for (int index = 0; index < sequence_length; index++) {
		int i = reversed ? sequence_length - 1 - index : index;
		struct ggml_tensor * I_cur = ggml_view_3d(ctx, I, I->ne[0], 1, I->ne[2], I->nb[0], I->nb[1], I->nb[1]*i);
		I_cur = ggml_sigmoid(ctx, ggml_add(ctx, I_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[1], h_0), biases[1])));

		struct ggml_tensor * F_cur = ggml_view_3d(ctx, F, F->ne[0], 1, F->ne[2], F->nb[0], F->nb[1], F->nb[1]*i);
		F_cur = ggml_sigmoid(ctx, ggml_add(ctx, F_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[3], h_0), biases[3])));

		struct ggml_tensor * G_cur = ggml_view_3d(ctx, G, G->ne[0], 1, G->ne[2], G->nb[0], G->nb[1], G->nb[1]*i);
		G_cur = ggml_tanh(ctx, ggml_add(ctx, G_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[5], h_0), biases[5])));

		struct ggml_tensor * O_cur = ggml_view_3d(ctx, O, O->ne[0], 1, O->ne[2], O->nb[0], O->nb[1], O->nb[1]*i);
		O_cur = ggml_sigmoid(ctx, ggml_add(ctx, O_cur, ggml_add(ctx, ggml_mul_mat(ctx, weights[7], h_0), biases[7])));

		c_0 = ggml_add(ctx, ggml_mul(ctx, F_cur, c_0), ggml_mul(ctx, I_cur, G_cur));
		h_0 = ggml_mul(ctx, ggml_tanh(ctx, c_0), O_cur);

		if (index == 0) {
			outputs = h_0;
		} else {
			outputs = reversed ? ggml_concat(ctx, h_0, outputs, 1) : ggml_concat(ctx, outputs, h_0, 1);
		}
		ggml_build_forward_expand(gf, outputs);
	}
	return outputs;
}

static struct ggml_tensor * build_ada_residual_conv(ggml_context * ctx, struct ggml_tensor * x, ada_residual_conv_block * block, struct ggml_tensor * style, struct ggml_tensor * sqrt_tensor) {
	struct ggml_tensor * cur = x;
	struct ggml_tensor * gamma;
	struct ggml_tensor * beta;

	gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->norm1_gamma, style), block->norm1_gamma_bias);
	beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->norm1_beta, style), block->norm1_beta_bias);
	cur   = ggml_norm(ctx, x, 0.00001);

	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
	// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
	cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_transpose(ctx, gamma)));
	cur = ggml_add(ctx, cur, ggml_transpose(ctx, beta));
	cur = ggml_leaky_relu(ctx, cur, 0.2f, false);

	if (block->pool) {
		cur = ggml_conv_transpose_1d_tts(ctx, block->pool, cur, 2, 1, 1, 1, cur->ne[1]);
		cur = ggml_add(ctx, cur, block->pool_bias);
	}

 	cur = ggml_conv_1d_tts(ctx, block->conv1, cur, 1, 1, 1);

	cur   = ggml_add(ctx, cur, block->conv1_bias);
	gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->norm2_gamma, style), block->norm2_gamma_bias);
	beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->norm2_beta, style), block->norm2_beta_bias);
	cur   = ggml_norm(ctx, cur, 0.00001);

	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
	// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
	cur = ggml_add(ctx, cur, ggml_mul(ctx, cur, ggml_transpose(ctx, gamma)));
	cur = ggml_add(ctx, cur, ggml_transpose(ctx, beta));
	cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
	cur = ggml_add(ctx, ggml_conv_1d_tts(ctx, block->conv2, cur, 1, 1, 1), block->conv2_bias);

	struct ggml_tensor * res = cur;
	cur = x;
	if (block->upsample) {
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
		if (block->pool) {
			cur = ggml_upscale_ext(ctx, cur, cur->ne[0], cur->ne[1]*2, cur->ne[2], cur->ne[3],GGML_SCALE_MODE_NEAREST);
		}
		cur = ggml_mul_mat(ctx, block->upsample, cur);
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
	}
	cur = ggml_div(ctx, ggml_add(ctx, res, cur), sqrt_tensor);
	return cur;
}

static struct ggml_tensor * build_kokoro_generator_res_block(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * style, kokoro_generator_residual_block * block) {
	struct ggml_tensor * cur;
	struct ggml_tensor * gamma;
	struct ggml_tensor * beta;
	struct ggml_tensor * inpl = x;
	for (int i = 0; i < block->convs1_weights.size(); i++) {
		gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_gamma_weights[i], style), block->adain1d_1_gamma_biases[i]);
		beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_1_beta_weights[i], style), block->adain1d_1_beta_biases[i]);
		cur   = ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, inpl, 0.00001)));

		// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
		cur   = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
		cur   = snake_1d(ctx, block->input_alphas[i], ggml_cont(ctx, ggml_transpose(ctx, cur)));

		cur   = ggml_add(ctx, ggml_conv_1d_tts(ctx, block->convs1_weights[i], cur, 1, block->conv1_paddings[i], block->conv1_dilations[i]), block->convs1_biases[i]);
		gamma = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_gamma_weights[i], style), block->adain1d_2_gamma_biases[i]);
		beta  = ggml_add(ctx, ggml_mul_mat(ctx, block->adain1d_2_beta_weights[i], style), block->adain1d_2_beta_biases[i]);
		cur   = ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, cur, 0.00001)));

		// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
		cur   = ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta)));

		cur   = snake_1d(ctx, block->output_alphas[i], cur);
		cur   = ggml_add(ctx, ggml_conv_1d_tts(ctx, block->convs2_weights[i], cur, 1, block->conv1_paddings[0], 1), block->convs2_biases[i]);
		inpl   = ggml_add(ctx, inpl, cur);
	}
	return inpl;
}

static struct ggml_tensor * build_noise_block(ggml_context * ctx, kokoro_noise_residual_block * block, struct ggml_tensor * x, struct ggml_tensor * style) {
	// This conv_1d seems replaceable with squeezed and transposed ggml_mul_mut, but s0 and p0 are dynamic
	ggml_tensor * cur = ggml_add(ctx, ggml_conv_1d_tts(ctx, block->input_conv, x, block->input_conv_stride, block->input_conv_padding, 1), block->input_conv_bias);
	return build_kokoro_generator_res_block(ctx, cur, style, block->res_block);
}

static struct ggml_tensor * build_sin_gen(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, int harmonic_num, int sequence_length, float voice_threshold, float sin_amp, float noise_std) {
	struct ggml_tensor * cur = ggml_mul(ctx, ggml_repeat(ctx, x, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x->ne[0], harmonic_num)), model->harmonic_sampling_norm);
	cur = ggml_mul(ctx, ggml_cumsum(ctx, ggml_mod(ctx, cur, 1.0f)), model->sampling_factor_scalar);
	cur = ggml_upscale_linear(ctx, cur, 300);
	struct ggml_tensor * upscaled = ggml_upscale_ext(ctx, x, x->ne[0]*300, x->ne[1], x->ne[2], x->ne[3],GGML_SCALE_MODE_NEAREST);

	kctx->uv_noise_data = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sequence_length*harmonic_num+4);
	ggml_set_input(kctx->uv_noise_data);

    struct ggml_tensor * fake = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sequence_length, harmonic_num, 2);

    // ggml doesn't support boolean tensors nor does it support greater than and roll ops. As a result, we represent these boolean tensors as 1.0 or 0.0 or simply perform
    // multiplications in place via a custom map.
    struct ggml_tensor * uv_noise = ggml_map_custom3(ctx, fake, upscaled, kctx->uv_noise_data, &uv_noise_compute, sequence_length, nullptr);


    struct ggml_tensor * noise = ggml_cont(ctx, ggml_view_2d(ctx, uv_noise, uv_noise->ne[0], uv_noise->ne[1], uv_noise->nb[1], uv_noise->nb[2]));
    struct ggml_tensor * uv = ggml_cont(ctx, ggml_view_2d(ctx, uv_noise, uv_noise->ne[0], uv_noise->ne[1], uv_noise->nb[1], 0));

	return ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_mul(ctx, ggml_sin(ctx, cur), uv), noise)));
}

static struct ggml_tensor * build_generator(ggml_context * ctx, kokoro_model * model, kokoro_context * kctx, struct ggml_tensor * x, struct ggml_tensor * style, struct ggml_tensor * f0_curve, kokoro_generator* generator, int sequence_length, struct ggml_tensor * window_sq_sum, ggml_cgraph * gf) {
	struct ggml_tensor * sing = build_sin_gen(ctx, model, kctx, f0_curve, model->harmonic_num + 1, f0_curve->ne[0] * 300, model->voice_threshold, model->sin_amp, model->noise_std);
	struct ggml_tensor * har = ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, generator->m_source_weight, sing), generator->m_source_bias));

	har = stft(ctx, ggml_cont(ctx, ggml_transpose(ctx, har)), generator->window, model->true_n_fft, model->stft_hop, true, true);

	// stft returns a vector of shape [nfft, frames, batch, 2] where the final shape (2) separates the magnitude and the phase
	// kokoro concatenates the n_fft from the magnitude and the phase together so we have to split them up and concatenate
	// along the n_fft axis
	struct ggml_tensor * mhar  = ggml_cont(ctx, ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], 0));
	struct ggml_tensor * phhar = ggml_cont(ctx, ggml_view_3d(ctx, har, har->ne[0], har->ne[1], har->ne[2], har->nb[1], har->nb[2], har->nb[3]));
	struct ggml_tensor * combined_har = ggml_cont(ctx, ggml_transpose(ctx, ggml_concat(ctx, mhar, phhar, 0)));

	struct ggml_tensor * cur = x;
	for (int i = 0; i < generator->ups.size(); i++) {
		cur = ggml_leaky_relu(ctx, cur, 0.1f, false);
		cur = ggml_add(ctx, ggml_conv_transpose_1d_tts(ctx, generator->ups[i]->upsample_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), generator->ups[i]->stride, generator->ups[i]->padding, 1, 0, 1), generator->ups[i]->upsample_bias);
		if (i == generator->ups.size() - 1) {
			// This is a hacky way of implementing the simple reflection padding used here.
			// In general, ggml should eventually be built to support expressive reflective padding but for such simple front padding this makes more sense.
			struct ggml_tensor * temp = ggml_cont(ctx, ggml_view_3d(ctx, cur, 1, cur->ne[1], cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[0]));
			cur = ggml_concat(ctx, temp, cur, 0);
		}
		struct ggml_tensor * x_source = build_noise_block(ctx, generator->noise_blocks[i], ggml_cont(ctx, combined_har), style);
		cur = ggml_add(ctx, cur, x_source);
		struct ggml_tensor * x = cur;
		for (int ii = 0; ii < model->n_kernels; ii++) {
			if (ii == 0) {
				cur = build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i*model->n_kernels+ii]);
			} else {
				cur = ggml_add(ctx, cur, build_kokoro_generator_res_block(ctx, x, style, generator->res_blocks[i*model->n_kernels+ii]));
			}
		}
		cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_div(ctx, cur, model->n_kernels_tensor)));
		ggml_build_forward_expand(gf, cur);
	}

	cur = ggml_leaky_relu(ctx, cur, 0.01f, false);
	cur = ggml_add(ctx, ggml_conv_1d_tts(ctx, generator->out_conv_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, model->out_conv_padding, 1), generator->out_conv_bias);

	struct ggml_tensor * spec = ggml_view_3d(ctx, cur, cur->ne[0], model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], 0);
	struct ggml_tensor * phase = ggml_view_3d(ctx, cur, cur->ne[0], cur->ne[1] - model->post_n_fft, cur->ne[2], cur->nb[1], cur->nb[2], cur->nb[1] * model->post_n_fft);
	phase = ggml_sin(ctx, phase);
	spec = ggml_exp(ctx, spec);

	cur = ggml_concat(ctx, spec, phase, 3); // istft expects the magnitude and phase concatenated after the batch;
	cur = istft(ctx, ggml_cont(ctx, ggml_transpose(ctx, cur)), window_sq_sum, generator->window, model->true_n_fft, model->stft_hop, true, true);
	ggml_set_name(cur, "after_res_gen");
	return cur;
}

static struct kokoro_generator_residual_block * build_res_block_from_file(gguf_context * meta, std::string base_config_key) {
	struct kokoro_generator_residual_block * grb = new struct kokoro_generator_residual_block;
	// these residual blocks always have 3 convolutional layers
	for (int i = 0; i < 3; i++) {
		grb->adain1d_1_gamma_weights.push_back(nullptr);
		grb->adain1d_2_gamma_weights.push_back(nullptr);
		grb->adain1d_1_gamma_biases.push_back(nullptr);
		grb->adain1d_2_gamma_biases.push_back(nullptr);
		grb->adain1d_1_beta_weights.push_back(nullptr);
		grb->adain1d_2_beta_weights.push_back(nullptr);
		grb->adain1d_1_beta_biases.push_back(nullptr);
		grb->adain1d_2_beta_biases.push_back(nullptr);
		grb->input_alphas.push_back(nullptr);
		grb->output_alphas.push_back(nullptr);
		grb->convs1_weights.push_back(nullptr);
		grb->convs1_biases.push_back(nullptr);
		grb->convs2_weights.push_back(nullptr);
		grb->convs2_biases.push_back(nullptr);
		int padding_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".padding").c_str());
		int dilation_key = gguf_find_key(meta, (base_config_key + "." + std::to_string(i) + ".dilation").c_str());
		if (padding_key == -1 || dilation_key == -1) {
			TTS_ABORT("Could not find dilation and padding for generator residual block at key, '%s.%d'.", base_config_key.c_str(), i);
		}
		grb->conv1_dilations.push_back(gguf_get_val_u32(meta, dilation_key));
		grb->conv1_paddings.push_back(gguf_get_val_u32(meta, padding_key));
	}
	return grb;
}

static struct kokoro_noise_residual_block * build_noise_block_from_file(gguf_context * meta, int index) {
	struct kokoro_noise_residual_block * nb = new struct kokoro_noise_residual_block;
	std::string base = "kokoro.decoder.generator.noise_blocks." + std::to_string(index);
	nb->res_block = build_res_block_from_file(meta, base + ".res_block");
	int stride_key = gguf_find_key(meta, (base + ".stride").c_str());
	int padding_key = gguf_find_key(meta, (base + ".padding").c_str());
	if (padding_key == -1 || stride_key == -1) {
		TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro noise block.");
	}
	nb->input_conv_stride = gguf_get_val_u32(meta, stride_key);
	nb->input_conv_padding = gguf_get_val_u32(meta, padding_key);
	return nb;
}

static struct kokoro_generator_upsample_block * kokoro_generator_upsample_block(gguf_context * meta, int index) {
	struct kokoro_generator_upsample_block * usb = new struct kokoro_generator_upsample_block;
	std::string base = "kokoro.decoder.generator.up_convs." + std::to_string(index);
	int stride_key = gguf_find_key(meta, (base + ".stride").c_str());
	int padding_key = gguf_find_key(meta, (base + ".padding").c_str());
	if (padding_key == -1 || stride_key == -1) {
		TTS_ABORT("both padding and stride keys must be assigned in order to initialize a kokoro upsample block.");
	}
	usb->stride = gguf_get_val_u32(meta, stride_key);
	usb->padding = gguf_get_val_u32(meta, padding_key);
	return usb;
}

size_t kokoro_model::max_gen_nodes() {
	return std::max<size_t>(8192, generation_node_counter*2);
}

size_t kokoro_model::max_duration_nodes() {
	return std::max<size_t>(8192, duration_node_counter*2);
}

void kokoro_model::post_load_assign() {
	size_t original_offset = offset;
	n_kernels_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	n_kernels_tensor->buffer = buf;
	n_kernels_tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
	size_t size = ggml_nbytes(n_kernels_tensor);
	float nker = (float) n_kernels;
	ggml_backend_tensor_set(n_kernels_tensor, &nker, 0, size);
	offset += size;

	sqrt_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	sqrt_tensor->buffer = buf;
   	sqrt_tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size = ggml_nbytes(sqrt_tensor);
    float sqrt2 = sqrtf(2.0f);
	ggml_backend_tensor_set(sqrt_tensor, &sqrt2, 0, size);
	offset += size;

	std::vector<float> data{};
	for (int l = 0; l < lstms.size(); l++) {
		lstm * rnn = lstms[l];
		const int32_t hidden_size = rnn->cells[0]->biases[0]->ne[0];
		data.resize(hidden_size);

		for (int i = 0; i < rnn->cells.size(); i++) {
			struct ggml_tensor * h = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
			struct ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
			h->buffer = buf;
    		h->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    		size_t size = ggml_nbytes(h);
			ggml_backend_tensor_set(h, data.data(), 0, size);
			ggml_format_name(h, "lstm%d_hidden", l);
    		offset += size;
			s->buffer = buf;
    		s->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
			ggml_backend_tensor_set(s, data.data(), 0, size);
			ggml_format_name(h, "lstm%d_state", l);
    		offset += size;
    		rnn->hidden.push_back(h);
    		rnn->states.push_back(s);
		}
		data.clear();
	}

	if (window == "hann") {
		std::vector<float> wdata;
		wdata.reserve(true_n_fft);
		hann_window(true_n_fft, wdata);
		decoder->generator->window = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, true_n_fft);
		decoder->generator->window->buffer = buf;
		decoder->generator->window->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
		size_t size = ggml_nbytes(decoder->generator->window);
		ggml_backend_tensor_set(decoder->generator->window, wdata.data(), 0, size);
    	ggml_set_name(decoder->generator->window, "stft_window");
    	offset += size;
    	wdata.clear();
	} else {
		TTS_ABORT("Window of type %s is not supported.", window.c_str());
	}

	harmonic_sampling_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, harmonic_num + 1);
	harmonic_sampling_norm->buffer = buf;
	harmonic_sampling_norm->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
	std::vector<float> hdata;
	hdata.reserve(harmonic_num + 1);
	for (int i = 0; i < harmonic_num + 1; i++) {
		hdata.push_back(((float)i + 1.0f) / sample_rate);
	}
	size_t hsize = ggml_nbytes(harmonic_sampling_norm);
	ggml_backend_tensor_set(harmonic_sampling_norm, hdata.data(), 0, hsize);
	hdata.clear();
	offset += hsize;

	sampling_factor_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	sampling_factor_scalar->buffer = buf;
   	sampling_factor_scalar->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t scsize = ggml_nbytes(sampling_factor_scalar);
    // while it might appear that the upsampling_rate could be used here, the interpolation rate (i.e. the upsampling scale) is actually independent in the kokoro model implementation.
    float sample_scalar = upsample_scale*2.0f*M_PI;
	ggml_backend_tensor_set(sampling_factor_scalar, &sample_scalar, 0, scsize);
	offset += scsize;
	post_load_tensor_bytes = 300 + offset - original_offset;
}

void kokoro_model::assign_lstm(lstm * rnn, std::string name, ggml_tensor * tensor) {
	std::vector<std::string> parts = split(name, ".");
	int i = std::stoi(parts[0]);
	int ii = std::stoi(parts[2]);
	if (parts[1] == "weights") {
		rnn->cells[i]->weights[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->weights[ii], tensor);
	} else if (parts[1] == "biases") {
		rnn->cells[i]->biases[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->biases[ii], tensor);
	} else if (parts[1] == "reverse_weights") {
		rnn->cells[i]->reverse_weights[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->reverse_weights[ii], tensor);
	} else if (parts[1] == "reverse_biases") {
		rnn->cells[i]->reverse_biases[ii] = ggml_dup_tensor(ctx, tensor);
		set_tensor(rnn->cells[i]->reverse_biases[ii], tensor);
	}
}

void kokoro_model::assign_weight(std::string name, ggml_tensor * tensor) {
	// all kokoro tensors are prepended by "kokoro" so lets trim that off and assign based on the module
	std::vector<std::string> parts = split(name, ".");
	if (parts.size() < 2) {
		return; // handle the null context tensor;
	}
	if (parts[1] == "albert") {
		assign_albert_weight(name.substr(7+parts[1].size()+1), tensor);
	} else if (parts[1] == "duration_predictor") {
		assign_duration_weight(name.substr(7+parts[1].size()+1), tensor);
	} else if (parts[1] == "text_encoder") {
		assign_text_encoder_weight(name.substr(7+parts[1].size()+1), tensor);
	} else if (parts[1] == "decoder") {
		assign_decoder_weight(name.substr(7+parts[1].size()+1), tensor);
	} else if (parts[1] == "voice_tensors") {
		voices[parts[2]] = ggml_dup_tensor(ctx, tensor);
		set_tensor(voices[parts[2]], tensor);
	}
}

void kokoro_model::assign_generator_weight(kokoro_generator * generator, std::string name, ggml_tensor * tensor) {
	if (name == "m_source_weight") {
		generator->m_source_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->m_source_weight, tensor);
	} else if (name == "m_source_bias") {
		generator->m_source_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->m_source_bias, tensor);
	} else if (name == "conv_post_weight") {
		generator->out_conv_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(generator->out_conv_weight, tensor);
	} else if (name == "conv_post_bias") {
		generator->out_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(generator->out_conv_bias, tensor);
	} else {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		if (parts[0] == "noise_blocks") {
			if (parts[2] == "conv_weight") {
				generator->noise_blocks[i]->input_conv = ggml_dup_tensor(ctx, tensor);
				set_tensor(generator->noise_blocks[i]->input_conv, tensor);
			} else if (parts[2] == "conv_bias") {
				generator->noise_blocks[i]->input_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
				set_tensor(generator->noise_blocks[i]->input_conv_bias, tensor);
			} else if (parts[2] == "resblock") {
				assign_gen_resblock(generator->noise_blocks[i]->res_block, name.substr(parts[0].size()+parts[1].size()+parts[2].size()+3), tensor);
			}
		} else if (parts[0] == "resblocks") {
			assign_gen_resblock(generator->res_blocks[i], name.substr(parts[0].size()+parts[1].size()+2), tensor);
		} else if (parts[0] == "ups") {
			if (parts[2] == "weight") {
				generator->ups[i]->upsample_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(generator->ups[i]->upsample_weight, tensor);
			} else if (parts[2] == "bias") {
				generator->ups[i]->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
				set_tensor(generator->ups[i]->upsample_bias, tensor);
			}
		}
	}
}

void kokoro_model::assign_gen_resblock(kokoro_generator_residual_block * block, std::string name, ggml_tensor * tensor) {
	std::vector<std::string> parts = split(name, ".");
	int i = std::stoi(parts[0]);
	if (parts[1] == "gamma1_weight") {
		block->adain1d_1_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_gamma_weights[i], tensor);
	} else if (parts[1] == "gamma2_weight") {
		block->adain1d_2_gamma_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_gamma_weights[i], tensor);
	} else if (parts[1] == "gamma1_bias") {
		block->adain1d_1_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_gamma_biases[i], tensor);
	} else if (parts[1] == "gamma2_bias") {
		block->adain1d_2_gamma_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_gamma_biases[i], tensor);
	} else if (parts[1] == "beta1_weight") {
		block->adain1d_1_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_beta_weights[i], tensor);
	} else if (parts[1] == "beta2_weight") {
		block->adain1d_2_beta_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_beta_weights[i], tensor);
	} else if (parts[1] == "beta1_bias") {
		block->adain1d_1_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_1_beta_biases[i], tensor);
	} else if (parts[1] == "beta2_bias") {
		block->adain1d_2_beta_biases[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->adain1d_2_beta_biases[i], tensor);
	} else if (parts[1] == "convs1_weight") {
		block->convs1_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->convs1_weights[i], tensor);
	} else if (parts[1] == "convs2_weight") {
		block->convs2_weights[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->convs2_weights[i], tensor);
	} else if (parts[1] == "convs1_bias") {
		block->convs1_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->convs1_biases[i], tensor);
	} else if (parts[1] == "convs2_bias") {
		block->convs2_biases[i] = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->convs2_biases[i], tensor);
	} else if (parts[1] == "alpha1") {
		block->input_alphas[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->input_alphas[i], tensor);
	} else if (parts[1] == "alpha2") {
		block->output_alphas[i] = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->output_alphas[i], tensor);
	}
}

/**
 * Removes the last axis, for cases where it's redundantly of length 1.
 * assert x.ndim == 3; numpy.squeeze(x, axis=-1)
 */
static ggml_tensor * squeeze_3d_2d_e0(ggml_context * ctx, ggml_tensor * x) {
	TTS_ASSERT(x->ne[0] == 1);
	TTS_ASSERT(ggml_is_contiguous(x));
	return ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);
}

void kokoro_model::assign_ada_res_block(ada_residual_conv_block * block, std::string name, ggml_tensor * tensor) {
	if (name == "norm1_gamma_weight") {
		block->norm1_gamma = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_gamma, tensor);
	} else if (name == "norm2_gamma_weight") {
		block->norm2_gamma = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_gamma, tensor);
	} else if (name == "norm1_gamma_bias") {
		block->norm1_gamma_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_gamma_bias, tensor);
	} else if (name == "norm2_gamma_bias") {
		block->norm2_gamma_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_gamma_bias, tensor);
	} else if (name == "norm1_beta_weight") {
		block->norm1_beta = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_beta, tensor);
	} else if (name == "norm2_beta_weight") {
		block->norm2_beta = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_beta, tensor);
	} else if (name == "norm1_beta_bias") {
		block->norm1_beta_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm1_beta_bias, tensor);
	} else if (name == "norm2_beta_bias") {
		block->norm2_beta_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->norm2_beta_bias, tensor);
	} else if (name == "conv1_weight") {
		block->conv1 = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->conv1, tensor);
	} else if (name == "conv2_weight") {
		block->conv2 = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->conv2, tensor);
	} else if (name == "conv1_bias") {
		block->conv1_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->conv1_bias, tensor);
	} else if (name == "conv2_bias") {
		block->conv2_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->conv2_bias, tensor);
	} else if (name == "pool_weight") {
		block->pool = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->pool, tensor);
	} else if (name == "pool_bias") {
		block->pool_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->pool_bias, tensor);
	} else if (name == "conv1x1_weight") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		block->upsample = ggml_dup_tensor(ctx, tensor);
		set_tensor(block->upsample, tensor);
	} else if (name == "conv1x1_bias") {
		block->upsample_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(block->upsample_bias, tensor);
	}
}

void kokoro_model::assign_decoder_weight(std::string name, ggml_tensor * tensor) {
	if (name == "f0_conv_weight") {
		decoder->f0_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->f0_conv, tensor);
	} else if (name == "f0_conv_bias") {
		decoder->f0_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->f0_conv_bias, tensor);
	} else if (name == "n_conv_weight") {
		decoder->n_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->n_conv, tensor);
	} else if (name == "n_conv_bias") {
		decoder->n_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->n_conv_bias, tensor);
	} else if (name == "asr_conv_weight") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		decoder->asr_conv = ggml_dup_tensor(ctx, tensor);
		set_tensor(decoder->asr_conv, tensor);
	} else if (name == "asr_conv_bias") {
		decoder->asr_conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(decoder->asr_conv_bias, tensor);
	} else if (has_prefix(name, "decoder_blocks")) {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		assign_ada_res_block(decoder->decoder_blocks[i], parts[2], tensor);
	} else if (has_prefix(name, "encoder_block")) {
		std::vector<std::string> parts = split(name, ".");
		assign_ada_res_block(decoder->encoder_block, parts[1], tensor);
	} else if (has_prefix(name, "generator")) {
		assign_generator_weight(decoder->generator, name.substr(10), tensor);
	}
}

void kokoro_model::assign_duration_weight(std::string name, ggml_tensor * tensor) {
	if (name == "encode") {
		prosody_pred->albert_encode = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->albert_encode , tensor);
	} else if (name == "encode_bias") {
		prosody_pred->albert_encode_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->albert_encode_bias, tensor);
	} else if (name == "duration_proj") {
		prosody_pred->duration_proj = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->duration_proj, tensor);
	} else if (name == "duration_proj_bias") {
		prosody_pred->duration_proj_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->duration_proj_bias, tensor);
	} else if (name == "n_proj_kernel") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		prosody_pred->n_proj_kernel = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->n_proj_kernel, tensor);
	} else if (name == "n_proj_bias") {
		prosody_pred->n_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(prosody_pred->n_proj_bias, tensor);
	} else if (name == "f0_proj_kernel") {
		tensor = squeeze_3d_2d_e0(ctx, tensor);
		prosody_pred->f0_proj_kernel = ggml_dup_tensor(ctx, tensor);
		set_tensor(prosody_pred->f0_proj_kernel, tensor);
	} else if (name == "f0_proj_bias") {
		prosody_pred->f0_proj_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
		set_tensor(prosody_pred->f0_proj_bias, tensor);
	} else {
		std::vector<std::string> parts = split(name, ".");
		if (parts[0] == "shared_lstm") {
			assign_lstm(prosody_pred->shared_lstm, name.substr(parts[0].size()+1), tensor);
		} else if (parts[0] == "duration_lstm") {
			assign_lstm(prosody_pred->duration_proj_lstm, name.substr(parts[0].size()+1), tensor);
		} else if (parts[0] == "f0_blocks") {
			int i = std::stoi(parts[1]);
			assign_ada_res_block(prosody_pred->f0_blocks[i], parts[2], tensor);
		} else if (parts[0] == "n_blocks") {
			int i = std::stoi(parts[1]);
			assign_ada_res_block(prosody_pred->n_blocks[i], parts[2], tensor);
		} else if (parts[0] == "layers") {
			int i = std::stoi(parts[1]);
			i = i / 2;
			if (parts[2] == "gamma_weight") {
				prosody_pred->layers[i]->ada_norm_gamma_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_gamma_weight , tensor);
			} else if (parts[2] == "gamma_bias") {
				prosody_pred->layers[i]->ada_norm_gamma_bias = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_gamma_bias , tensor);
			} else if (parts[2] == "beta_weight") {
				prosody_pred->layers[i]->ada_norm_beta_weight = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_beta_weight , tensor);
			} else if (parts[2] == "beta_bias") {
				prosody_pred->layers[i]->ada_norm_beta_bias = ggml_dup_tensor(ctx, tensor);
				set_tensor(prosody_pred->layers[i]->ada_norm_beta_bias , tensor);
			} else if (parts[2] == "lstm") {
				assign_lstm(prosody_pred->layers[i]->rnn, name.substr(parts[0].size()+parts[1].size()+parts[2].size()+3), tensor);
			}
		}
	}
}

void kokoro_model::assign_text_encoder_weight(std::string name, ggml_tensor * tensor) {
	if (name == "embedding_weight") {
		text_encoder->embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(text_encoder->embd, tensor);
	} else if (has_prefix(name, "lstm")) {
		assign_lstm(text_encoder->out_lstm, name.substr(5), tensor);
	} else if (has_prefix(name, "layers")) {
		std::vector<std::string> parts = split(name, ".");
		int i = std::stoi(parts[1]);
		if (parts[2] == "gamma") {
			text_encoder->conv_layers[i]->norm_gamma = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->norm_gamma, tensor);
		} else if (parts[2] == "beta") {
			text_encoder->conv_layers[i]->norm_beta = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->norm_beta, tensor);
		} else if (parts[2] == "weight") {
			text_encoder->conv_layers[i]->conv_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(text_encoder->conv_layers[i]->conv_weight, tensor);
		} else if (parts[2] == "bias") {
			text_encoder->conv_layers[i]->conv_bias = ggml_dup_tensor(ctx, ggml_transpose(ctx, tensor));
			set_tensor(text_encoder->conv_layers[i]->conv_bias, tensor);
		}
	}
}

void kokoro_model::assign_albert_weight(std::string name, ggml_tensor * tensor) {
	if (name == "embd") {
		embd_hidden = ggml_dup_tensor(ctx, tensor);
		set_tensor(embd_hidden, tensor);
	} else if (name == "embd_bias") {
		embd_hidden_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(embd_hidden_bias, tensor);
	} else if (name == "token_embd") {
		token_embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(token_embd, tensor);
	} else if (name == "position_embd") {
		position_embd = ggml_dup_tensor(ctx, tensor);
		set_tensor(position_embd, tensor);
	} else if (name == "norm") {
		input_norm_weight = ggml_dup_tensor(ctx, tensor);
		set_tensor(input_norm_weight, tensor);
	} else if (name == "norm_bias") {
		input_norm_bias = ggml_dup_tensor(ctx, tensor);
		set_tensor(input_norm_bias, tensor);
	} else if (name == "token_type_embd") {
		static_token_type_values = ggml_dup_tensor(ctx, tensor);
		set_tensor(static_token_type_values, tensor);
	} else if (has_prefix(name, "layer")) {
		std::vector<std::string> parts = split(name, '.');
		int i = std::stoi(parts[1]);
		if (parts[2] == "ffn") {
			layers[i]->ffn = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn, tensor);
		} else if (parts[2] == "ffn_bias") {
			layers[i]->ffn_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_bias, tensor);
		} else if (parts[2] == "ffn_out") {
			layers[i]->ffn_out = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_out, tensor);
		} else if (parts[2] == "ffn_out_bias") {
			layers[i]->ffn_out_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->ffn_out_bias, tensor);
		} else if (parts[2] == "attn_norm") {
			layers[i]->layer_output_norm_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->layer_output_norm_weight, tensor);
		} else if (parts[2] == "attn_norm_bias") {
			layers[i]->layer_output_norm_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->layer_output_norm_bias, tensor);
		} else if (parts[2] == "q") {
			layers[i]->q = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->q, tensor);
		} else if (parts[2] == "k") {
			layers[i]->k = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->k, tensor);
		} else if (parts[2] == "v") {
			layers[i]->v = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->v, tensor);
		} else if (parts[2] == "o") {
			layers[i]->o = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->o, tensor);
		}  else if (parts[2] == "q_bias") {
			layers[i]->q_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->q_bias, tensor);
		} else if (parts[2] == "k_bias") {
			layers[i]->k_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->k_bias, tensor);
		} else if (parts[2] == "v_bias") {
			layers[i]->v_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->v_bias, tensor);
		} else if (parts[2] == "o_bias") {
			layers[i]->o_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->o_bias, tensor);
		} else if (parts[2] == "ffn_norm") {
			layers[i]->attn_norm_weight = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->attn_norm_weight, tensor);
		} else if (parts[2] == "ffn_norm_bias") {
			layers[i]->attn_norm_bias = ggml_dup_tensor(ctx, tensor);
			set_tensor(layers[i]->attn_norm_bias, tensor);
		}
	}
}

lstm * kokoro_model::prep_lstm() {
	lstm * rnn = new lstm;
	lstm_cell * cell = new lstm_cell;
	for (int i = 0; i < 8; i++) {
		cell->weights.push_back(nullptr);
		cell->biases.push_back(nullptr);
		cell->reverse_weights.push_back(nullptr);
		cell->reverse_biases.push_back(nullptr);
	}
	rnn->cells.push_back(cell);
	rnn->bidirectional = true;
	lstms.push_back(rnn);
	return rnn;
}

void kokoro_model::prep_layers(gguf_context * meta) {
	prosody_pred = new duration_predictor;
	prosody_pred->shared_lstm = prep_lstm();
	prosody_pred->duration_proj_lstm = prep_lstm();
	text_encoder = new kokoro_text_encoder;
	decoder = new kokoro_decoder;
	decoder->generator = new kokoro_generator;
	decoder->encoder_block = new ada_residual_conv_block;
	text_encoder->out_lstm = prep_lstm();

	for (int i = 0; i < n_layers; i++) {
		layers.push_back(new albert_layer);
	}

	for (int i = 0; i < f0_n_blocks; i++) {
		ada_residual_conv_block * f0 = new ada_residual_conv_block;
		ada_residual_conv_block * n = new ada_residual_conv_block;
		prosody_pred->f0_blocks.push_back(f0);
		prosody_pred->n_blocks.push_back(n);
	}

	for (int i = 0; i < n_duration_prediction_layers; i++) {
		duration_predictor_layer* dpl = new duration_predictor_layer;
		dpl->rnn = prep_lstm();
		prosody_pred->layers.push_back(dpl);
	}

	for (int i = 0; i < n_decoder_blocks; i++) {
		decoder->decoder_blocks.push_back(new ada_residual_conv_block);
	}

	for (int i = 0; i < n_noise_blocks; i++) {
		struct kokoro_noise_residual_block * nb = build_noise_block_from_file(meta, i);
		decoder->generator->noise_blocks.push_back(nb);
	}

	for (int i = 0; i < n_upsamples; i++) {
		struct kokoro_generator_upsample_block * ub = kokoro_generator_upsample_block(meta, i);
		decoder->generator->ups.push_back(ub);
	}

	for (int i = 0; i < n_res_blocks; i++) {
		struct kokoro_generator_residual_block* rb = build_res_block_from_file(meta, "kokoro.decoder.generator.res_blocks." + std::to_string(i));
		decoder->generator->res_blocks.push_back(rb);
	}

	for (int i = 0; i < n_conv_layers; i++) {
		text_encoder->conv_layers.push_back(new kokoro_text_encoder_conv_layer);
	}
}

void kokoro_model::prep_constants(gguf_context * meta) {
	// get constants for the Albert duration prediction model
	int context_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.context_length");
    if (context_size_key != -1) {
        max_context_length = gguf_get_val_u32(meta, context_size_key);;
    }

    int vocab_size_key = gguf_find_key(meta, "kokoro.tokenizer.vocab_size");
    if (vocab_size_key != -1) {
        vocab_size = gguf_get_val_u32(meta, vocab_size_key);
    }

    int hidden_size_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.hidden_size");
    if (hidden_size_key != -1) {
        hidden_size = gguf_get_val_u32(meta, hidden_size_key);
    }

    int attn_heads_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.attn_heads");
    if (attn_heads_key != -1) {
        n_attn_heads = gguf_get_val_u32(meta, attn_heads_key);
        head_size = (uint32_t) hidden_size / n_attn_heads;
    }

    int albert_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.layers");
    if (albert_layers_key != -1) {
        n_layers = gguf_get_val_u32(meta, albert_layers_key);
    }

    int recurrence_key = gguf_find_key(meta, "kokoro.duration_predictor.albert.recurrence");
    if (recurrence_key != -1) {
        n_recurrence = gguf_get_val_u32(meta, recurrence_key);
    }

    int duration_hidden_key = gguf_find_key(meta, "kokoro.duration_predictor.hidden_size");
    if (duration_hidden_key != -1) {
        duration_hidden_size = gguf_get_val_u32(meta, duration_hidden_key);
    }

    int up_sampling_factor_key = gguf_find_key(meta, "kokoro.decoder.generator.up_sampling_factor");
    if (up_sampling_factor_key != -1) {
        up_sampling_factor = gguf_get_val_u32(meta, up_sampling_factor_key);
    }

	int f0_n_blocks_key = gguf_find_key(meta, "kokoro.duration_predictor.f0_n_blocks");
    if (f0_n_blocks_key != -1) {
        f0_n_blocks = gguf_get_val_u32(meta, f0_n_blocks_key);
    }

   	int duration_pred_layers_key = gguf_find_key(meta, "kokoro.duration_predictor.layers");
    if (duration_pred_layers_key != -1) {
        n_duration_prediction_layers = gguf_get_val_u32(meta, duration_pred_layers_key);
    }

	// get text and decoding configuration for generation
	int n_conv_layers_key = gguf_find_key(meta, "kokoro.text_encoder.layers");
    if (n_conv_layers_key != -1) {
        n_conv_layers = gguf_get_val_u32(meta, n_conv_layers_key);
    }

   	int n_kernels_key = gguf_find_key(meta, "kokoro.decoder.generator.kernels");
    if (n_kernels_key != -1) {
        n_kernels = gguf_get_val_u32(meta, n_kernels_key);
    }

    int n_upsamples_key = gguf_find_key(meta, "kokoro.decoder.generator.upsamples");
    if (n_upsamples_key != -1) {
        n_upsamples = gguf_get_val_u32(meta, n_upsamples_key);
    }

    int n_decoder_blocks_key = gguf_find_key(meta, "kokoro.decoder.generator.layers");
    if (n_decoder_blocks_key != -1) {
        n_decoder_blocks = gguf_get_val_u32(meta, n_decoder_blocks_key);
    }

    int out_conv_padding_key = gguf_find_key(meta, "kokoro.decoder.generator.padding");
    if (out_conv_padding_key != -1) {
        out_conv_padding = gguf_get_val_u32(meta, out_conv_padding_key);
    }

    int n_fft_key = gguf_find_key(meta, "kokoro.decoder.generator.n_fft");
    if (n_fft_key != -1) {
        true_n_fft = gguf_get_val_u32(meta, n_fft_key);
        post_n_fft = (uint32_t) true_n_fft / 2 + 1;
    }

    int stft_hop_key = gguf_find_key(meta, "kokoro.decoder.generator.hop");
    if (stft_hop_key != -1) {
        stft_hop = gguf_get_val_u32(meta, stft_hop_key);
    }
}

kokoro_ubatch kokoro_duration_runner::build_worst_case_batch() {
	kokoro_ubatch batch;
	batch.n_tokens = model->max_context_length;
	return batch;
}

struct ggml_cgraph * kokoro_duration_runner::build_kokoro_duration_graph(kokoro_ubatch & batch) {
    init_build();
    // This '110000' number is coming from the number of nodes necessary for the longest possible sequence computed by of the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 110000, false);

    struct ggml_tensor * voice = model->voices[kctx->voice];
    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    if (!model->static_token_types) {
    	kctx->token_types = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    	ggml_set_input(kctx->token_types);
    }

    kctx->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->positions);

    inpL = build_albert_inputs(ctx, model, kctx->inp_tokens, kctx->positions, kctx->token_types);
    ggml_set_name(inpL, "albert_embeddings");
    cur = inpL;

    struct ggml_tensor * KQ_mask_dec = build_albert_attn_mask(ctx, kctx, batch);

    for (int r = 0; r < model->n_recurrence; r++) {
    	for (int l = 0; l < model->n_layers; l++) {
	        struct ggml_tensor * residual = cur ;
	        struct ggml_tensor * attn_out;

	        // self-attention
	        {
	            struct ggml_tensor * Qcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->q, cur), model->layers[l]->q_bias);
	            struct ggml_tensor * Kcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->k, cur), model->layers[l]->k_bias);
	            struct ggml_tensor * Vcur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->v, cur), model->layers[l]->v_bias);

				Qcur = ggml_reshape_3d(ctx, Qcur, model->head_size, model->n_attn_heads, batch.n_tokens);
	            Kcur = ggml_reshape_3d(ctx, Kcur, model->head_size, model->n_attn_heads, batch.n_tokens);

	            struct ggml_tensor * q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
	            struct ggml_tensor * k = ggml_cont(ctx, ggml_permute(ctx, Kcur, 0, 2, 1, 3));
	            struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

	            kq = ggml_soft_max_ext(ctx, kq, KQ_mask_dec, model->scale, 0.0f);

	            struct ggml_tensor * v = ggml_cont_3d(ctx, ggml_transpose(ctx, Vcur), batch.n_tokens, model->head_size, model->n_attn_heads);
	            struct ggml_tensor * kqv = ggml_mul_mat(ctx, kq, v);
	            struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 2, 0, 1, 3);
	            attn_out = ggml_cont_2d(ctx, kqv_merged, model->hidden_size, batch.n_tokens);
	            attn_out = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->o, attn_out), model->layers[l]->o_bias);
	        }
	        cur = ggml_add(ctx, attn_out, residual);
	        cur = build_albert_norm(ctx, cur, model->layers[l]->attn_norm_weight, model->layers[l]->attn_norm_bias);

	        struct ggml_tensor * residualffn = cur;

	        // ffn
	        {
	        	cur = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn, cur), model->layers[l]->ffn_bias));
	        	cur = ggml_add(ctx, ggml_mul_mat(ctx, model->layers[l]->ffn_out, cur), model->layers[l]->ffn_out_bias);
	        }

			cur = ggml_add(ctx, cur, residualffn);
			cur = build_albert_norm(ctx, cur, model->layers[l]->layer_output_norm_weight, model->layers[l]->layer_output_norm_bias);
	    }
        ggml_build_forward_expand(gf, cur);
    }

    // duration / prosody prediction
    cur = ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->albert_encode, cur), model->prosody_pred->albert_encode_bias);

	struct ggml_tensor * style_half = ggml_cont(ctx, ggml_view_1d(ctx, voice, voice->ne[0]/2, voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]));

	cur = ggml_concat(ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);

    for (auto l : model->prosody_pred->layers) {
    	cur = build_lstm(ctx, cur, l->rnn, batch.n_tokens, gf);

    	struct ggml_tensor * gamma = ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_gamma_weight, style_half), l->ada_norm_gamma_bias);
    	struct ggml_tensor * beta = ggml_add(ctx, ggml_mul_mat(ctx, l->ada_norm_beta_weight, style_half), l->ada_norm_beta_bias);

    	cur = ggml_norm(ctx, cur, 0.00001);

    	// The addition between gamma * x and x is performed here because ggml doesn't support scalar multiplication without initializing the scalars in advance.
		// An optimal remedy to this would be to increment the gamma bias above by one when preparing the gguf file for the model.
    	cur = ggml_add(ctx, ggml_add(ctx, cur, ggml_mul(ctx, cur, gamma)), beta);
    	cur = ggml_concat(ctx, cur, ggml_repeat(ctx, style_half, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_half->ne[0], cur->ne[1])), 0);
    }

    struct ggml_tensor * d = ggml_cont(ctx, cur);
    ggml_set_name(d, "duration_hidden_states");
    ggml_build_forward_expand(gf, d);

    struct ggml_tensor * len;
    cur = build_lstm(ctx, cur, model->prosody_pred->duration_proj_lstm, batch.n_tokens, gf);
    cur = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, model->prosody_pred->duration_proj, cur), model->prosody_pred->duration_proj_bias));
    // If we were to support speed we would add a constant tensor for the speed and divide here.
    len = ggml_round(ctx, ggml_sum_rows(ctx, cur));
    len = ggml_clamp(ctx, ggml_round(ctx, ggml_sum_rows(ctx, cur)), 1.0f, 50.0f);

    ggml_build_forward_expand(gf, len);

    free_build();

    return gf;
}

void kokoro_duration_runner::prepare_post_load() {
    auto batch = build_worst_case_batch();
    auto gf = build_kokoro_duration_graph(batch);
    kctx->prep_schedule(gf);
}

void kokoro_duration_runner::set_inputs(kokoro_ubatch & batch) {
	ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0, batch.n_tokens*ggml_element_size(kctx->inp_tokens));
	uint32_t * positions_d = nullptr;
    positions_d = (uint32_t *) kctx->positions->data;
    float * attn_d = nullptr;
    attn_d = (float *) kctx->attn_mask->data;
	for (uint32_t i = 0; i < batch.n_tokens; i++) {
        positions_d[i] =  i;
        for (uint32_t ii = 0; ii < batch.n_tokens; ii++) {
        	attn_d[i*batch.n_tokens + ii] = 0.0f; // Kokoro doesn't use causal attention as it isnt an autoregressive generative model;
        }
    }
}

void kokoro_duration_runner::run(kokoro_ubatch & batch) {
    ggml_backend_sched_reset(kctx->sched);

    size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    size_t new_size = model->max_context_length * (model->duration_hidden_size + model->style_half_size) * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
	    if (kctx->buf_output) {
	        ggml_backend_buffer_free(kctx->buf_output);
	        kctx->buf_output = nullptr;
	        kctx->logits = nullptr;
	    }
	    kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
	}

    prev_size = kctx->buf_len_output ? ggml_backend_buffer_get_size(kctx->buf_len_output) : 0;
    new_size = model->max_context_length * sizeof(float);

    if (!kctx->buf_len_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_len_output);
            kctx->buf_len_output = nullptr;
            kctx->lens = nullptr;
        }

        kctx->buf_len_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }


    batch.resp->hidden_states = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);
    batch.resp->lengths = (float *) ggml_backend_buffer_get_base(kctx->buf_len_output);
    ggml_backend_buffer_clear(kctx->buf_len_output, 0);

    struct ggml_cgraph * gf = NULL;
    gf = build_kokoro_duration_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * lens = gf->nodes[gf->n_nodes - 1];
    // the reused duration hidden states are computed before a node chunk which has a size that is sequence length dependent
    struct ggml_tensor * hidden_states = gf->nodes[gf->n_nodes - 22 - 52 * batch.n_tokens];
    ggml_backend_sched_alloc_graph(kctx->sched, gf);

    set_inputs(batch);

    ggml_backend_sched_graph_compute_async(kctx->sched, gf);

    kctx->get_ggml_node_data(lens, batch.resp->lengths, batch.n_tokens*sizeof(float), kctx->buf_len_output);
    kctx->get_ggml_node_data(hidden_states, batch.resp->hidden_states, batch.n_tokens*(model->duration_hidden_size+model->style_half_size)*sizeof(float));

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(kctx->sched);
    batch.resp->n_outputs = batch.n_tokens;
}

kokoro_ubatch kokoro_runner::build_worst_case_batch() {
	kokoro_ubatch batch;
	batch.n_tokens = model->max_context_length;
	batch.resp = new kokoro_duration_response;
	batch.resp->n_outputs = model->max_context_length;
	kctx->total_duration = model->max_context_length * model->max_duration_per_token;
	kctx->sequence_length = model->max_context_length;
	std::vector<float> lengths;
	lengths.reserve(model->max_context_length);
	for (int i = 0; i < model->max_context_length; i++) {
		lengths.push_back(50.0f);
	}
	batch.resp->lengths = lengths.data();
	return batch;
}

struct ggml_cgraph * kokoro_runner::build_kokoro_graph(kokoro_ubatch & batch) {
    init_build();
    // This '570000' number is coming from the number of nodes necessary for the longest possible sequence computed by the graph.
    // While it may be possible to precompute this by determining the longest possible duration against he maximum context length of the model,
    // it is not easily performed given that nodes do not necessarily line up predictably with the number of tensors in the model or its submodels.
    // In order to side step this problem I computed the graph and determined the size in advance and use that constant value here.
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 570000, false);

    struct ggml_tensor * voice = model->voices[kctx->voice];
    struct ggml_tensor * style_half = ggml_view_1d(ctx, voice, voice->ne[0]/2, voice->ne[0] / 2 * voice->nb[0] + (batch.n_tokens - 3) * voice->nb[1]);
    struct ggml_tensor * cur;

    kctx->inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
    ggml_set_input(kctx->inp_tokens);

    kctx->duration_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kctx->total_duration, kctx->sequence_length);
    ggml_set_input(kctx->duration_mask);

    kctx->duration_pred = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model->duration_hidden_size + model->style_half_size, kctx->sequence_length);
    ggml_set_input(kctx->duration_pred);

    // seeing as we are setting the inputs for these, we shouldn't need to perform tranpositions here
    cur = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_mask)), ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_pred)));
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

    cur = build_lstm(ctx, cur, model->prosody_pred->shared_lstm, cur->ne[1], gf);


    struct ggml_tensor * f0_curve = cur;
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    for (auto block : model->prosody_pred->f0_blocks) {
    	f0_curve = build_ada_residual_conv(ctx, f0_curve, block, style_half, model->sqrt_tensor);
    }
    f0_curve = ggml_cont(ctx, ggml_transpose(ctx, f0_curve));
    f0_curve = ggml_mul_mat(ctx, model->prosody_pred->f0_proj_kernel, f0_curve);
    f0_curve = squeeze_3d_2d_e0(ctx, f0_curve);
    f0_curve = ggml_add(ctx, f0_curve, model->prosody_pred->f0_proj_bias);
    ggml_set_name(f0_curve, "f0_out");

    struct ggml_tensor * n = cur;
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
    for (auto block : model->prosody_pred->n_blocks) {
		n = build_ada_residual_conv(ctx, n, block, style_half, model->sqrt_tensor);
    }
    n = ggml_cont(ctx, ggml_transpose(ctx, n));
	n = ggml_mul_mat(ctx, model->prosody_pred->n_proj_kernel, n);
	n = squeeze_3d_2d_e0(ctx, n);
	n = ggml_add(ctx, n, model->prosody_pred->n_proj_bias);
	ggml_set_name(n, "n_out");
	ggml_build_forward_expand(gf, n);

	// kokoro text encoding;
	struct ggml_tensor * asr;
	//struct ggml_tensor * embd;
	{
		cur = ggml_get_rows(ctx, model->text_encoder->embd, kctx->inp_tokens);

		for (auto l : model->text_encoder->conv_layers) {
			cur = ggml_cont(ctx, ggml_transpose(ctx, ggml_add(ctx, ggml_conv_1d_tts(ctx, l->conv_weight, ggml_cont(ctx, ggml_transpose(ctx, cur)), 1, 2, 1), l->conv_bias)));
			cur = ggml_norm(ctx, cur, 0.00001);
			cur = ggml_add(ctx, ggml_mul(ctx, cur, l->norm_gamma), l->norm_beta);
			cur = ggml_leaky_relu(ctx, cur, 0.2f, false);
		}

		cur = build_lstm(ctx, cur, model->text_encoder->out_lstm, kctx->sequence_length, gf);
		asr = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, cur)), ggml_cont(ctx, ggml_transpose(ctx, kctx->duration_mask)));
	}

	// decoding and generation prep
	struct ggml_tensor * asr_res;
	struct ggml_tensor * f0;
	struct ggml_tensor * n_base;
	struct ggml_tensor * style_half2 = ggml_view_1d(ctx, voice, voice->ne[0]/2, (batch.n_tokens - 3) * voice->nb[1]);

	{
		f0 = ggml_add(ctx, ggml_conv_1d_tts(ctx, model->decoder->f0_conv, f0_curve, 2, 1, 1), model->decoder->f0_conv_bias);
		n_base = ggml_add(ctx, ggml_conv_1d_tts(ctx, model->decoder->n_conv, n, 2, 1, 1), model->decoder->n_conv_bias);
		cur = ggml_concat(ctx, ggml_concat(ctx, ggml_cont(ctx, ggml_transpose(ctx, asr)), f0, 1), n_base, 1);
		cur = build_ada_residual_conv(ctx, cur, model->decoder->encoder_block, style_half2, model->sqrt_tensor);
		ggml_build_forward_expand(gf, cur);

		asr_res = ggml_mul_mat(ctx, model->decoder->asr_conv, asr);
		asr_res = ggml_add(ctx, asr_res, ggml_transpose(ctx, model->decoder->asr_conv_bias));

		asr_res = ggml_cont(ctx, ggml_transpose(ctx, asr_res));
		for (auto l : model->decoder->decoder_blocks) {
			cur = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, cur, asr_res, 1), f0, 1), n_base, 1 );
			cur = build_ada_residual_conv(ctx, cur, l, style_half2, model->sqrt_tensor);
			ggml_build_forward_expand(gf, cur);
		}
		cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
	}

	kctx->window_sq_sum = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kctx->total_duration*model->up_sampling_factor);
	ggml_set_input(kctx->window_sq_sum);

	// run generation
	cur = build_generator(ctx, model, kctx, cur, style_half2, f0_curve, model->decoder->generator, (int)kctx->sequence_length, kctx->window_sq_sum, gf);
    ggml_build_forward_expand(gf, cur);
    free_build();
    return gf;
}

void kokoro_runner::prepare_post_load() {
	model->post_load_assign();
	drunner->prepare_post_load();
    auto batch = build_worst_case_batch();
    auto gf = build_kokoro_graph(batch);
    kctx->prep_schedule(gf);
    free(batch.resp);
}

void kokoro_runner::set_inputs(kokoro_ubatch & batch, uint32_t total_size) {
	random_uniform_gen(total_size * model->up_sampling_factor * (model->harmonic_num + 1), ((float*)kctx->uv_noise_data->data) + 4);
    ((float*) kctx->uv_noise_data->data)[0] = model->voice_threshold;
    ((float*) kctx->uv_noise_data->data)[1] = model->noise_std;
    ((float*) kctx->uv_noise_data->data)[2] = model->sin_amp;
    ((float*) kctx->uv_noise_data->data)[3] = model->sin_amp / 3.0f;
	compute_window_squared_sum(model->true_n_fft, model->stft_hop, total_size*model->up_sampling_factor/model->stft_hop, (float*) kctx->window_sq_sum->data, (float*) model->decoder->generator->window->data);
	kctx->sequence_length = batch.n_tokens;
	kctx->total_duration = total_size;
	ggml_backend_tensor_set(kctx->inp_tokens, batch.input_tokens, 0, batch.n_tokens*ggml_element_size(kctx->inp_tokens));
	ggml_backend_tensor_set(kctx->duration_pred, batch.resp->hidden_states, 0, batch.n_tokens*(model->duration_hidden_size + model->style_half_size)*ggml_element_size(kctx->duration_pred));
	float * d = nullptr;
	float running = 0;
    d = (float *) kctx->duration_mask->data;
	for (uint32_t i = 0; i < batch.n_tokens; i++) {
		float next_running = running + batch.resp->lengths[i];
		for (uint32_t ii = 0; ii < total_size; ii++) {
			d[i*total_size+ii] = ii >= running && ii < next_running ? 1.0f : 0.0f;
		}
		running = next_running;
    }
}

void kokoro_runner::run(kokoro_ubatch & batch, tts_response * outputs) {
	batch.resp = new kokoro_duration_response;
	drunner->run(batch);

	ggml_backend_sched_reset(kctx->sched);

    const size_t prev_size = kctx->buf_output ? ggml_backend_buffer_get_size(kctx->buf_output) : 0;
    uint32_t total_length = 0;
    for (int i = 0; i < batch.resp->n_outputs; i++) {
    	total_length += (uint32_t) batch.resp->lengths[i];
    }
    const size_t new_size = total_length * model->up_sampling_factor * sizeof(float);

    if (!kctx->buf_output || prev_size < new_size) {
        if (kctx->buf_output) {
            ggml_backend_buffer_free(kctx->buf_output);
            kctx->buf_output = nullptr;
            kctx->logits = nullptr;
        }
        kctx->buf_output = ggml_backend_buft_alloc_buffer(kctx->backend_cpu_buffer, new_size);
    }

    outputs->data = (float *) ggml_backend_buffer_get_base(kctx->buf_output);
    ggml_backend_buffer_clear(kctx->buf_output, 0);

    kctx->sequence_length = batch.n_tokens;
	kctx->total_duration = total_length;

    struct ggml_cgraph * gf = NULL;
    gf = build_kokoro_graph(batch);

    // the output is always the last tensor in the graph
    struct ggml_tensor * output = gf->nodes[gf->n_nodes - 1];

    ggml_backend_sched_alloc_graph(kctx->sched, gf);

    set_inputs(batch, total_length);

    ggml_backend_sched_graph_compute_async(kctx->sched, gf);

    kctx->get_ggml_node_data(output, outputs->data, new_size);

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    ggml_backend_sched_reset(kctx->sched);
    outputs->n_outputs = total_length*model->up_sampling_factor;
    free(batch.resp);
    return;
}

void kokoro_runner::assign_weight(std::string name, ggml_tensor * tensor) {
	model->assign_weight(name, tensor);
}

/*
 * #tokenize_chunks is used to split up a larger than max context size (512) token prompt into discrete
 * blocks for generation. This solution, in accordance with Kokoro's pyTorch implementation, splits
 * the prompt by sentence when possible (this can result in slower inference but generally produces cleaner
 * speech). If a disinct sentence is too long, then it splits at the nearest space.
 */
std::vector<std::vector<uint32_t>> kokoro_runner::tokenize_chunks(std::vector<std::string> clauses) {
	std::vector<std::vector<uint32_t>> chunks;
	for (auto clause : clauses) {
		clause = strip(clause);
		if (clause.empty()) {
			continue;
		}
		std::vector<uint32_t> tokens;
		tokens.push_back(model->bos_token_id);
		tokenizer->tokenize(clause, tokens);
		// if there are more clause tokens than the max context length then try to split by space tokens.
		// To be protective, split mid-word when there are no spaces (this should never happen).
		if (tokens.size() > model->max_context_length - 2) {
			// we skip the first token here becuase it is the bos token.
			int last_space_token = 1;
			int last_split = 1;
			for (int i = 1; i < tokens.size(); i++) {
				if (tokens[i] == model->space_token_id) {
					last_space_token = i;
				}
				if ((i - last_split) + chunks.back().size() >= model->max_context_length - 1) {
					if (last_space_token > last_split) {
						std::vector<uint32_t> portion = { model->bos_token_id };
						portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + last_space_token);
						portion.push_back(model->eos_token_id);
						chunks.push_back(portion);
						last_split = last_space_token;
					} else {
						std::vector<uint32_t> portion = { model->bos_token_id };
						portion.insert(portion.end(), tokens.begin() + last_split, tokens.begin() + i + 1);
						portion.push_back(model->eos_token_id);
						chunks.push_back(portion);
						last_split = i + 1;
					}
				}
			}
			if (last_split + 1 < tokens.size()) {
				std::vector<uint32_t> portion = { model->bos_token_id };
				portion.insert(portion.end(), tokens.begin() + last_split, tokens.end());
				portion.push_back(model->eos_token_id);
				chunks.push_back(portion);
			}
		} else {
			tokens.push_back(model->eos_token_id);
			chunks.push_back(tokens);
		}
	}
	return chunks;
}

//kcpp hacked a quick replace fn
static void kokoro_str_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

int kokoro_runner::generate(std::string prompt, struct tts_response * response, std::string voice, std::string voice_code) {
	if (model->voices.find(voice) == model->voices.end()) {
		fprintf(stdout,"\nFailed to find Kokoro voice '%s' aborting.\n", voice.c_str());
		return -1;
    } else {
    	// if the language changed then we should change the phonemization voice
    	if (phmzr->mode == ESPEAK && kctx->voice[0] != voice[0]) {
    		if (voice_code.empty()) {
    			voice_code = get_espeak_id_from_kokoro_voice(voice);
    		}
    		update_voice(voice_code);
    	}
        kctx->voice = voice;
        drunner->kctx->voice = voice;
    }
    // replace all non-sentence terminating characters with '--' which espeak will treat as a pause.
    // We preserve the other punctuation for cleaner chunking pre-tokenization
    prompt = replace_any(prompt, ";:", "--");
    prompt = replace_any(prompt, "\n", "--");
	kokoro_str_replace_all(prompt,"’","'");
	kokoro_str_replace_all(prompt,"Mr. ","Mister ");
	prompt = std::regex_replace(prompt, std::regex("(\\w)([.!?]) "), "$1$2, ");
	kokoro_str_replace_all(prompt," - "," -- ");
	kokoro_str_replace_all(prompt,"he's ","he is ");
	kokoro_str_replace_all(prompt,"'s ","s ");
	kokoro_str_replace_all(prompt,"n't ","nt ");
	kokoro_str_replace_all(prompt,"*"," ");
  	std::string phonemized_prompt = phmzr->text_to_phonemes(prompt);
	// printf("\nRESULT: %s\n",phonemized_prompt.c_str());

  	// Kokoro users a utf-8 single character tokenizer so if the size of the prompt is smaller than the max context length without the
  	// beginning of sentence and end of sentence tokens then we can compute it all at once.
  	if (phonemized_prompt.size() < model->max_context_length - 2) {
  		// we preserved punctuation and Kokoro interprets these tokens as end of sentence tokens, so we have to remove them for all-at-once compute.
  		phonemized_prompt = strip(replace_any(phonemized_prompt, ".!?", ""));
  		if (phonemized_prompt.empty()) {
  			return 0;
  		}
		std::vector<uint32_t> tokens;
		tokens.push_back(model->bos_token_id);
		tokenizer->tokenize(phonemized_prompt, tokens);
		tokens.push_back(model->eos_token_id);
		kokoro_ubatch batch;
		batch.n_tokens = tokens.size();
		batch.input_tokens = tokens.data();
		run(batch, response);
  	} else {
  		// TODO: determine the performance to memory trade off in using a batched compute approach verse this chunking approach.
  		// This approach is likely to be slower than a batched approach, but given the already huge memory overhead of Kokoro's graph it
  		// might be preferable to use this chunking approach.
  		std::vector<std::string> clauses = split(phonemized_prompt, ".!?");
  		for (auto tokens : tokenize_chunks(clauses)) {
			kokoro_ubatch batch;
			batch.n_tokens = tokens.size();
			batch.input_tokens = tokens.data();
			struct tts_response * partial = new tts_response;
			run(batch, partial);
			append_to_response(response, partial);
		}
  	}
  	return 0;
}

std::vector<std::string> kokoro_runner::list_voices() {
	std::vector<std::string> voices;
	voices.reserve(model->voices.size());
	for (auto voice : model->voices) {
		voices.push_back(voice.first);
	}
	return voices;
}


std::string get_espeak_id_from_kokoro_voice(std::string voice) {
	return !voice.empty() && KOKORO_LANG_TO_ESPEAK_ID.find(voice[0]) != KOKORO_LANG_TO_ESPEAK_ID.end() ? KOKORO_LANG_TO_ESPEAK_ID[voice[0]] : "gmw/en-US";
}

struct kokoro_duration_context * build_new_duration_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_duration_context * kctx = new kokoro_duration_context(model, n_threads);

    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_duration_nodes()*5 + ggml_graph_overhead_custom(model->max_duration_nodes()*5, false));
    return kctx;
}


struct kokoro_context * build_new_kokoro_context(struct kokoro_model * model, int n_threads, bool use_cpu) {
    kokoro_context * kctx = new kokoro_context(model, n_threads);
    kctx->backend_cpu = ggml_backend_cpu_init();
    kctx->set_threads();
    kctx->build_schedule();
    kctx->buf_compute_meta.resize(ggml_tensor_overhead()*model->max_gen_nodes()*30 + ggml_graph_overhead_custom(model->max_gen_nodes()*30, false));
    return kctx;
}
