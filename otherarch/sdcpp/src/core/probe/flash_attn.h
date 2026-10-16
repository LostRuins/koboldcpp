#ifndef __SD_CORE_PROBE_FLASH_ATTN_H__
#define __SD_CORE_PROBE_FLASH_ATTN_H__

#include "ggml-backend.h"

bool sd_probe_flash_attention(ggml_backend_t backend, int n_threads);

#endif  // __SD_CORE_PROBE_FLASH_ATTN_H__
