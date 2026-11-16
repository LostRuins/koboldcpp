#ifndef __SD_RUNTIME_GENERATION_PROGRESS_H__
#define __SD_RUNTIME_GENERATION_PROGRESS_H__

#include "stable-diffusion.h"

namespace sd {

    class GenerationProgress {
        sd_generation_progress_cb_t callback_ = nullptr;
        void* data_                           = nullptr;
        int sample_pass_                      = 0;
        const char* sample_detail_            = "";

    public:
        void set_callback(sd_generation_progress_cb_t callback, void* data) {
            callback_      = callback;
            data_          = data;
            sample_pass_   = 0;
            sample_detail_ = "";
        }

        void report(const char* stage, const char* detail = "", int64_t completed = 0, int64_t total = 0, const char* unit = "") const {
            if (callback_ != nullptr) {
                const sd_generation_progress_t progress{stage, detail, unit, sample_pass_, completed, total};
                callback_(&progress, data_);
            }
        }

        void begin_sample(int64_t total, const char* detail) {
            ++sample_pass_;
            sample_detail_ = detail;
            report_sample(0, total);
        }

        void report_sample(int64_t completed, int64_t total) const {
            report("sampling", sample_detail_, completed, total, "steps");
        }
    };

}  // namespace sd

#endif  // __SD_RUNTIME_GENERATION_PROGRESS_H__
