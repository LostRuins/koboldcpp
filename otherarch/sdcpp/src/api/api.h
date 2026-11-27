#ifndef SD_INTERNAL_API_H
#define SD_INTERNAL_API_H

#include <exception>
#include <new>

#include "stable-diffusion.h"

class StableDiffusionGGML;

struct sd_ctx_t {
    StableDiffusionGGML* sd = nullptr;
};

namespace sd::api {
    void clear_error();
    sd_status_t error(sd_status_t status, const char* key, const char* reason);

    template <typename Function>
    sd_status_t call(Function&& function, sd_status_t failure = SD_STATUS_INVALID_ARGUMENT) noexcept {
        clear_error();
        try {
            return function();
        } catch (const std::bad_alloc&) {
            return error(SD_STATUS_OUT_OF_MEMORY, nullptr, "out of memory");
        } catch (const std::exception& exception) {
            return error(failure, nullptr, exception.what());
        } catch (...) {
            return error(failure, nullptr, "unexpected exception");
        }
    }
}

#endif  // SD_INTERNAL_API_H
