#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::assets {

// Opens a PyTorch `.bin` checkpoint (a ZIP archive wrapping a pickled state dict) as a
// tensor source. Only the restricted subset produced by torch.save of a flat
// OrderedDict[str, Tensor] or a single root tensor (exposed as "tensor") is supported.
// The pickle is walked with a whitelist of
// globals so no adapter code is ever executed. Tensor storages must be uncompressed.
// Tensors that are views into a shared storage are read from their storage offset;
// non-contiguous tensors (e.g. a saved transpose) are rejected.
std::shared_ptr<const TensorSource> open_torch_bin_tensor_source(const std::filesystem::path & path);

}  // namespace engine::assets
