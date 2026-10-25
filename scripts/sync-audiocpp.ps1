param(
    [Parameter(Mandatory = $true)]
    [string] $SourceRoot,

    [ValidateSet("supertonic", "yue2", "all")]
    [string[]] $Component = @("all"),

    [switch] $ModelSourcesOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$vendorRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "otherarch/audiocpp/upstream"))
$sourceRoot = [System.IO.Path]::GetFullPath($SourceRoot)

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "CMakeLists.txt") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $sourceRoot "include/engine/framework") -PathType Container)) {
    throw "SourceRoot does not look like an Audio.cpp checkout: $sourceRoot"
}

function Copy-AudioCppFile([string] $relativePath) {
    $source = Join-Path $sourceRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Audio.cpp source file is missing: $relativePath"
    }

    $destination = Join-Path $vendorRoot $relativePath
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

function Copy-AudioCppTree([string] $relativePath) {
    $source = Join-Path $sourceRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Audio.cpp source directory is missing: $relativePath"
    }

    Get-ChildItem -LiteralPath $source -Recurse -File | ForEach-Object {
        $sourcePrefix = $sourceRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
        $relativeFile = $_.FullName.Substring($sourcePrefix.Length)
        Copy-AudioCppFile $relativeFile
    }
}

$foundationFiles = @(
    "LICENSE",
    "external/cJSON/LICENSE",
    "external/cJSON/cJSON.c",
    "external/cJSON/cJSON.h",
    "src/framework/assets/resource_bundle.cpp",
    "src/framework/assets/tensor_source.cpp",
    "src/framework/assets/weight_adapter.cpp",
    "src/framework/assets/weight_adapter_source.cpp",
    "src/framework/assets/weight_norm_folding.cpp",
    "src/framework/model_spec/package.cpp",
    "src/framework/model_spec/options.cpp",
    "src/framework/model_spec/schema.cpp",
    "src/framework/core/backend.cpp",
    "src/framework/core/deferred_tensor_writer.cpp",
    "src/framework/core/execution_context.cpp",
    "src/framework/core/load_progress.cpp",
    "src/framework/core/module.cpp",
    "src/framework/runtime/artifacts.cpp",
    "src/framework/runtime/cache.cpp",
    "src/framework/runtime/graph_executor.cpp",
    "src/framework/runtime/model.cpp",
    "src/framework/runtime/options.cpp",
    "src/framework/runtime/session.cpp",
    "src/framework/runtime/session_base.cpp",
    "src/framework/runtime/task_vocabulary.cpp",
    "src/framework/runtime/workspace.cpp",
    "src/framework/io/binary.cpp",
    "src/framework/io/config.cpp",
    "src/framework/io/filesystem.cpp",
    "src/framework/io/json.cpp",
    "src/framework/io/safetensors.cpp",
    "src/framework/io/text.cpp",
    "src/framework/debug/profiler.cpp",
    "src/framework/debug/trace.cpp",
    "src/framework/text/chunking.cpp",
    "src/framework/text/unicode_normalization.cpp",
    "src/framework/text/unicode_nfkd_data.inc",
    "src/framework/modules/activation_modules.cpp",
    "src/framework/modules/conv_modules.cpp",
    "src/framework/modules/linear_module.cpp",
    "src/framework/modules/lookup_modules.cpp",
    "src/framework/modules/norm_modules.cpp",
    "src/framework/modules/packed_linear_weights.cpp",
    "src/framework/modules/primitive_modules.cpp",
    "src/framework/modules/streaming_conv_modules.cpp",
    "src/framework/modules/structural_modules.cpp",
    "src/framework/modules/tensor_layout_utils.h"
)

$supertonicFiles = @(
    "model_specs/supertonic.json",
    "src/models/supertonic/assets.cpp",
    "src/models/supertonic/loader.cpp",
    "src/models/supertonic/runtime.cpp",
    "src/models/supertonic/session.cpp",
    "src/models/supertonic/tokenizer_text.cpp"
)

# YuE2 is intentionally a separate import tier. Its framework closure is larger
# than Supertonic's, but it still avoids Audio.cpp's server, CLI, converters,
# unrelated models, bundled GGML, sentencepiece, and YAML dependencies.
$yue2Files = @(
    "model_specs/yue2.json",
    "src/framework/core/attention_fallback.cpp",
    "src/framework/runtime/bounded_static_kv_decode.cpp",
    "src/framework/runtime/graph_executor.cpp",
    "src/framework/runtime/graph_optimizer.cpp",
    "src/framework/runtime/greedy_causal_decoder.cpp",
    "src/framework/runtime/host_ops.cpp",
    "src/framework/runtime/kv_cache.cpp",
    "src/framework/modules/attention/grouped_query_attention.cpp",
    "src/framework/modules/attention/attention_internal.h",
    "src/framework/modules/module_internal.h",
    "src/framework/modules/attention/positional_encoding.cpp",
    "src/framework/modules/attention/scaled_dot_product_attention.cpp",
    "src/framework/modules/positional_modules.cpp",
    "src/framework/modules/optimizations/fast_kv_modules.cpp",
    "src/framework/modules/transformers/causal_decoder.cpp",
    "src/framework/modules/transformers/causal_decoder_runtime.cpp",
    "src/framework/modules/transformers/decoder.cpp",
    "src/framework/tokenizers/llama_bpe.cpp",
    "src/framework/codecs/oobleck_audio_vae_runtime.cpp",
    "src/framework/sampling/greedy_decode.cpp",
    "src/framework/sampling/hf_sampler.cpp",
    "src/framework/sampling/torch_random.cpp",
    "src/models/yue2/ar_runtime.cpp",
    "src/models/yue2/assets.cpp",
    "src/models/yue2/lora.cpp",
    "src/models/yue2/nar_runtime.cpp",
    "src/models/yue2/pipeline.cpp",
    "src/models/yue2/request.cpp",
    "src/models/yue2/session.cpp",
    "src/models/yue2/tokenizer_text.cpp",
    "src/models/yue2/types.cpp"
)

$selected = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($name in $Component) {
    if ($name -eq "all") {
        [void] $selected.Add("supertonic")
        [void] $selected.Add("yue2")
    } else {
        [void] $selected.Add($name)
    }
}

# Headers are inexpensive and are kept as one upstream-owned API surface. The
# model-only mode is useful after local compatibility patches have been applied
# to an already synchronized foundation.
if (-not $ModelSourcesOnly) {
    Copy-AudioCppTree "include/engine/framework"
    foreach ($relativePath in $foundationFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("supertonic")) {
    Copy-AudioCppTree "include/engine/models/supertonic"
    foreach ($relativePath in $supertonicFiles) {
        Copy-AudioCppFile $relativePath
    }
}

if ($selected.Contains("yue2")) {
    Copy-AudioCppTree "include/engine/models/yue2"
    # Reuse KoboldCpp's newer unicode implementation. Audio.cpp's tokenizer
    # only contributes the BPE core and its matching public declarations.
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.cpp"
    Copy-AudioCppFile "external/llama_tokenizer/bpe-core.h"
    Copy-AudioCppFile "external/llama_tokenizer/unicode.h"
    foreach ($relativePath in $yue2Files) {
        Copy-AudioCppFile $relativePath
    }
}

$generatedDirectory = Join-Path $vendorRoot "generated"
New-Item -ItemType Directory -Force -Path $generatedDirectory | Out-Null
$modelSpecEntries = ""
Get-ChildItem -LiteralPath (Join-Path $vendorRoot "model_specs") -Filter "*.json" -File |
    Sort-Object Name |
    ForEach-Object {
        $family = $_.BaseName
        $json = [System.IO.File]::ReadAllText($_.FullName)
        if ($json.Contains(")AUDIOCPP_SPEC`"")) {
            throw "Model spec contains the reserved C++ raw-string delimiter: $($_.FullName)"
        }
        $modelSpecEntries += "    {`"$family`", R`"AUDIOCPP_SPEC($json)AUDIOCPP_SPEC`"},`n"
    }
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path $generatedDirectory "model_specs.inc"),
    $modelSpecEntries,
    $utf8WithoutBom)

Write-Host "Synchronized Audio.cpp components [$($selected -join ', ')] into $vendorRoot"
Write-Host "Reapply and review files in otherarch/audiocpp/patches after every synchronization."
