# Scalar libwebp 1.6.0 source manifest for KoboldCpp.
# Keep this explicit: platform SIMD sources, tools and examples are intentionally
# not part of the vendored build.

TINYLIBWEBP_ROOT := vendor/tinylibwebp

TINYLIBWEBP_SOURCES := \
	$(TINYLIBWEBP_ROOT)/src/dec/alpha_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/buffer_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/frame_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/idec_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/io_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/quant_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/tree_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/vp8_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/vp8l_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dec/webp_dec.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/alpha_processing.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/cost.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/cpu.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/dec.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/dec_clip_tables.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/enc.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/filters.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/lossless.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/lossless_enc.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/rescaler.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/ssim.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/upsampling.c \
	$(TINYLIBWEBP_ROOT)/src/dsp/yuv.c \
	$(TINYLIBWEBP_ROOT)/src/enc/alpha_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/analysis_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/backward_references_cost_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/backward_references_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/config_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/cost_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/filter_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/frame_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/histogram_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/iterator_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/near_lossless_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/picture_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/picture_csp_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/picture_psnr_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/picture_rescale_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/picture_tools_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/predictor_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/quant_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/syntax_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/token_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/tree_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/vp8l_enc.c \
	$(TINYLIBWEBP_ROOT)/src/enc/webp_enc.c \
	$(TINYLIBWEBP_ROOT)/src/mux/anim_encode.c \
	$(TINYLIBWEBP_ROOT)/src/mux/muxedit.c \
	$(TINYLIBWEBP_ROOT)/src/mux/muxinternal.c \
	$(TINYLIBWEBP_ROOT)/src/mux/muxread.c \
	$(TINYLIBWEBP_ROOT)/src/utils/bit_reader_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/bit_writer_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/color_cache_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/filters_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/huffman_encode_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/huffman_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/palette.c \
	$(TINYLIBWEBP_ROOT)/src/utils/quant_levels_dec_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/quant_levels_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/random_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/rescaler_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/thread_utils.c \
	$(TINYLIBWEBP_ROOT)/src/utils/utils.c \
	$(TINYLIBWEBP_ROOT)/sharpyuv/sharpyuv.c \
	$(TINYLIBWEBP_ROOT)/sharpyuv/sharpyuv_cpu.c \
	$(TINYLIBWEBP_ROOT)/sharpyuv/sharpyuv_csp.c \
	$(TINYLIBWEBP_ROOT)/sharpyuv/sharpyuv_dsp.c \
	$(TINYLIBWEBP_ROOT)/sharpyuv/sharpyuv_gamma.c

TINYLIBWEBP_HEADERS := \
	$(wildcard $(TINYLIBWEBP_ROOT)/src/dec/*.h) \
	$(wildcard $(TINYLIBWEBP_ROOT)/src/dsp/*.h) \
	$(wildcard $(TINYLIBWEBP_ROOT)/src/enc/*.h) \
	$(wildcard $(TINYLIBWEBP_ROOT)/src/mux/*.h) \
	$(wildcard $(TINYLIBWEBP_ROOT)/src/utils/*.h) \
	$(wildcard $(TINYLIBWEBP_ROOT)/src/webp/*.h) \
	$(wildcard $(TINYLIBWEBP_ROOT)/sharpyuv/*.h)

OBJS_TINYLIBWEBP := $(TINYLIBWEBP_SOURCES:.c=.o)
LIB_TINYLIBWEBP := $(TINYLIBWEBP_ROOT)/libtinylibwebp.a

# Do not inherit native CPU tuning: these objects are deliberately baseline C.
TINYLIBWEBP_CFLAGS := $(filter-out -march=% -mtune=% -mcpu=% -mfpu=% -mavx% -msse% -mssse% -mfma -mf16c -funsafe-math-optimizations,$(CFLAGS))
TINYLIBWEBP_CFLAGS += -I./$(TINYLIBWEBP_ROOT) -I./$(TINYLIBWEBP_ROOT)/src -DHAVE_CONFIG_H -DWEBP_DISABLE_SIMD
