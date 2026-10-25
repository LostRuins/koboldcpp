set(AUDIOCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/upstream)

# The synchronized tree is deliberately selective. Treat it as one upstream
# integration unit so adding another imported model does not require another
# KoboldCpp target or a model-specific source manifest.
file(GLOB_RECURSE AUDIOCPP_SOURCES CONFIGURE_DEPENDS
    ${AUDIOCPP_ROOT}/src/*.cpp
    ${AUDIOCPP_ROOT}/external/*.cpp
    ${AUDIOCPP_ROOT}/external/*.c
)

add_library(audiocpp STATIC ${AUDIOCPP_SOURCES})
target_include_directories(audiocpp
    PUBLIC
        ${AUDIOCPP_ROOT}/include
        ${AUDIOCPP_ROOT}/generated
    PRIVATE
        ${AUDIOCPP_ROOT}/external/cJSON
        ${AUDIOCPP_ROOT}/external/llama_tokenizer
)
target_compile_features(audiocpp PUBLIC cxx_std_17)
target_compile_definitions(audiocpp PRIVATE CJSON_HIDE_SYMBOLS)
target_link_libraries(audiocpp PRIVATE ggml)
set_target_properties(audiocpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
