#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen3_tts {

class CoreMLCodePredictor {
public:
    CoreMLCodePredictor();
    ~CoreMLCodePredictor();

    CoreMLCodePredictor(const CoreMLCodePredictor &) = delete;
    CoreMLCodePredictor & operator=(const CoreMLCodePredictor &) = delete;

    bool load(const std::string & model_dir, int32_t n_steps);
    void unload();

    bool is_loaded() const;
    const std::string & get_error() const;

    bool predict_step(int32_t step_idx,
                      const float * seq_embd,
                      int32_t seq_len,
                      int32_t hidden_size,
                      std::vector<float> & logits_out);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

} // namespace qwen3_tts

