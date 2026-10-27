# Portable C-only subset of libwebp 1.6.0. Keep this list synchronized with
# sources.mk; architecture-specific implementations are intentionally omitted.

set(TINYLIBWEBP_SOURCES
    src/dec/alpha_dec.c
    src/dec/buffer_dec.c
    src/dec/frame_dec.c
    src/dec/idec_dec.c
    src/dec/io_dec.c
    src/dec/quant_dec.c
    src/dec/tree_dec.c
    src/dec/vp8_dec.c
    src/dec/vp8l_dec.c
    src/dec/webp_dec.c
    src/dsp/alpha_processing.c
    src/dsp/cost.c
    src/dsp/cpu.c
    src/dsp/dec.c
    src/dsp/dec_clip_tables.c
    src/dsp/enc.c
    src/dsp/filters.c
    src/dsp/lossless.c
    src/dsp/lossless_enc.c
    src/dsp/rescaler.c
    src/dsp/ssim.c
    src/dsp/upsampling.c
    src/dsp/yuv.c
    src/enc/alpha_enc.c
    src/enc/analysis_enc.c
    src/enc/backward_references_cost_enc.c
    src/enc/backward_references_enc.c
    src/enc/config_enc.c
    src/enc/cost_enc.c
    src/enc/filter_enc.c
    src/enc/frame_enc.c
    src/enc/histogram_enc.c
    src/enc/iterator_enc.c
    src/enc/near_lossless_enc.c
    src/enc/picture_enc.c
    src/enc/picture_csp_enc.c
    src/enc/picture_psnr_enc.c
    src/enc/picture_rescale_enc.c
    src/enc/picture_tools_enc.c
    src/enc/predictor_enc.c
    src/enc/quant_enc.c
    src/enc/syntax_enc.c
    src/enc/token_enc.c
    src/enc/tree_enc.c
    src/enc/vp8l_enc.c
    src/enc/webp_enc.c
    src/mux/anim_encode.c
    src/mux/muxedit.c
    src/mux/muxinternal.c
    src/mux/muxread.c
    src/utils/bit_reader_utils.c
    src/utils/bit_writer_utils.c
    src/utils/color_cache_utils.c
    src/utils/filters_utils.c
    src/utils/huffman_encode_utils.c
    src/utils/huffman_utils.c
    src/utils/palette.c
    src/utils/quant_levels_dec_utils.c
    src/utils/quant_levels_utils.c
    src/utils/random_utils.c
    src/utils/rescaler_utils.c
    src/utils/thread_utils.c
    src/utils/utils.c
    sharpyuv/sharpyuv.c
    sharpyuv/sharpyuv_cpu.c
    sharpyuv/sharpyuv_csp.c
    sharpyuv/sharpyuv_dsp.c
    sharpyuv/sharpyuv_gamma.c
)

add_library(tinylibwebp STATIC ${TINYLIBWEBP_SOURCES})
target_compile_features(tinylibwebp PUBLIC c_std_11)
target_compile_definitions(tinylibwebp PRIVATE HAVE_CONFIG_H WEBP_DISABLE_SIMD)
target_include_directories(tinylibwebp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(tinylibwebp PROPERTIES
    C_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
    target_link_libraries(tinylibwebp PUBLIC m)
endif()

add_executable(tinylibwebp_smoke EXCLUDE_FROM_ALL tests/anim_smoke.c)
target_link_libraries(tinylibwebp_smoke PRIVATE tinylibwebp)

add_custom_target(tinylibwebp-smoke
    COMMAND $<TARGET_FILE:tinylibwebp_smoke>
    DEPENDS tinylibwebp_smoke
    COMMENT "Running animated WebP smoke test")
