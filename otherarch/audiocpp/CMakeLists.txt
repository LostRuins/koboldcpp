set(AUDIOCPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/upstream)

# The synchronized tree is deliberately selective. Treat it as one upstream
# integration unit so adding another imported model does not require another
# KoboldCpp target or a model-specific source manifest.
file(GLOB_RECURSE AUDIOCPP_SOURCES CONFIGURE_DEPENDS
    ${AUDIOCPP_ROOT}/src/*.cpp
    ${AUDIOCPP_ROOT}/external/*.cpp
    ${AUDIOCPP_ROOT}/external/*.cc
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
        ${AUDIOCPP_ROOT}/external/libyaml/include
        ${AUDIOCPP_ROOT}/external/sentencepiece
        ${AUDIOCPP_ROOT}/external/sentencepiece/src
        ${AUDIOCPP_ROOT}/external/sentencepiece/src/builtin_pb
        ${AUDIOCPP_ROOT}/external/sentencepiece/third_party
        ${AUDIOCPP_ROOT}/external/sentencepiece/third_party/protobuf-lite
)
target_compile_features(audiocpp PUBLIC cxx_std_17)
target_compile_definitions(audiocpp PRIVATE
    CJSON_HIDE_SYMBOLS
    _USE_INTERNAL_STRING_VIEW
    HAVE_PTHREAD=1
    _POSIX_C_SOURCE=200809L
    YAML_DECLARE_STATIC=1
    YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2
    YAML_VERSION_PATCH=5
    YAML_VERSION_STRING="0.2.5"
)
target_link_libraries(audiocpp PRIVATE ggml)
set_target_properties(audiocpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
