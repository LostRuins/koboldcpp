#include "torch_legacy_io.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pickle_io.h"
#include "util.h"

// torch.save format background:
//
//   - Before PyTorch 1.6.0, torch.save used this legacy non-zip format by
//     default.
//   - Since PyTorch 1.6.0, torch.save defaults to an uncompressed ZIP64 archive
//     containing data.pkl, data/, version, and, since PyTorch 2.1.0, byteorder.
//   - The old format can still be produced explicitly with:
//       torch.save(obj, path, _use_new_zipfile_serialization=False)
//
// Whether obj is a state_dict or a whole nn.Module does not change the outer
// container format selected by torch.save. It changes the pickled object inside:
//
//   - state_dict: usually an OrderedDict[str, Tensor]. pickle_io.cpp supports a
//     restricted subset of this layout because tensor metadata and raw storages
//     can be recovered without executing pickle callables.
//   - whole module/checkpoint object: arbitrary Python object graph. This may
//     require importing user classes and executing pickle GLOBAL/REDUCE rebuild
//     logic, so it is intentionally not supported here.
//
// Legacy non-zip PyTorch files are not a single pickle object:
//
//   1. pickle object: PyTorch legacy magic number
//   2. pickle object: legacy protocol version, expected to be 1001
//   3. pickle object: sys_info metadata, ignored by this reader
//   4. pickle object: state_dict metadata, parsed by pickle_io.cpp
//   5. pickle object: serialized storage key list, skipped here
//   6. raw storage data payloads
//      - PyTorch writes storages after the pickles, ordered by storage key
//      - each storage has an 8-byte legacy storage header followed by raw bytes
static constexpr size_t LEGACY_STORAGE_HEADER_SIZE = 8;

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

static std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    static const char* hex = "0123456789ABCDEF";
    std::string result;
    result.reserve(bytes.size() * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i > 0) {
            result.push_back('-');
        }
        result.push_back(hex[(bytes[i] >> 4) & 0x0F]);
        result.push_back(hex[bytes[i] & 0x0F]);
    }
    return result;
}

static bool is_probably_tar_file(const std::vector<uint8_t>& header) {
    return header.size() >= 262 &&
           header[257] == 'u' &&
           header[258] == 's' &&
           header[259] == 't' &&
           header[260] == 'a' &&
           header[261] == 'r';
}

static std::string torch_legacy_diagnostics(const std::string& file_path, const std::vector<uint8_t>& buffer) {
    if (!ends_with(file_path, ".pt") && !ends_with(file_path, ".pth")) {
        return "";
    }
    if (buffer.empty()) {
        return "unsupported PyTorch file '" + file_path + "': empty file";
    }

    size_t short_len = std::min<size_t>(buffer.size(), 32);
    std::vector<uint8_t> short_header(buffer.begin(), buffer.begin() + short_len);
    const bool raw_pickle = buffer[0] == 0x80;
    const bool tar_file   = is_probably_tar_file(buffer);

    std::string message = "unsupported PyTorch file '" + file_path + "': first bytes " +
                          bytes_to_hex(short_header) +
                          ", raw_pickle=" + (raw_pickle ? "true" : "false") +
                          ", tar=" + (tar_file ? "true" : "false");
    if (raw_pickle) {
        message += "; raw pickle did not match the restricted state_dict layouts currently supported";
    } else if (tar_file) {
        message += "; legacy tar PyTorch checkpoints are not supported yet";
    }
    return message;
}

bool read_torch_legacy_file(const std::string& file_path,
                            std::vector<TensorStorage>& tensor_storages,
                            std::string* error) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        set_error(error, "failed to open '" + file_path + "'");
        return false;
    }

    file.seekg(0, file.end);
    size_t file_size = (size_t)file.tellg();
    file.seekg(0, file.beg);
    if (file_size == 0) {
        set_error(error, "empty file '" + file_path + "'");
        return false;
    }

    std::vector<uint8_t> buffer(file_size);
    file.read((char*)buffer.data(), file_size);
    if (!file) {
        set_error(error, "failed to read '" + file_path + "'");
        return false;
    }

    auto finalize_tensor_offsets = [&](size_t storage_data_offset,
                                       const std::unordered_map<std::string, uint64_t>& legacy_storage_map) -> bool {
        if (storage_data_offset > file_size) {
            return false;
        }

        std::vector<std::string> storage_keys;
        storage_keys.reserve(legacy_storage_map.size());
        for (const auto& [storage_key, _] : legacy_storage_map) {
            storage_keys.push_back(storage_key);
        }
        std::sort(storage_keys.begin(), storage_keys.end());

        std::unordered_map<std::string, uint64_t> storage_offsets;
        uint64_t current_offset = storage_data_offset;
        for (const auto& storage_key : storage_keys) {
            auto it = legacy_storage_map.find(storage_key);
            if (it == legacy_storage_map.end()) {
                return false;
            }
            if (current_offset + LEGACY_STORAGE_HEADER_SIZE + it->second > file_size) {
                return false;
            }
            storage_offsets[storage_key] = current_offset + LEGACY_STORAGE_HEADER_SIZE;
            current_offset += LEGACY_STORAGE_HEADER_SIZE + it->second;
        }

        for (auto& tensor_storage : tensor_storages) {
            if (tensor_storage.storage_key.empty()) {
                continue;
            }

            auto it_offset = storage_offsets.find(tensor_storage.storage_key);
            auto it_size   = legacy_storage_map.find(tensor_storage.storage_key);
            if (it_offset == storage_offsets.end() || it_size == legacy_storage_map.end()) {
                return false;
            }

            uint64_t base_offset    = it_offset->second;
            uint64_t storage_nbytes = it_size->second;
            uint64_t tensor_nbytes  = tensor_storage.nbytes_to_read();
            if (tensor_storage.offset + tensor_nbytes > storage_nbytes) {
                return false;
            }

            tensor_storage.offset = base_offset + tensor_storage.offset;
            tensor_storage.storage_key.clear();
        }

        return true;
    };

    auto parse_state_dict_at = [&](size_t state_dict_offset, size_t state_dict_size, size_t* storage_data_offset) -> bool {
        tensor_storages.clear();
        std::unordered_map<std::string, uint64_t> legacy_storage_map;
        if (!parse_torch_state_dict_pickle(buffer.data() + state_dict_offset,
                                           state_dict_size,
                                           tensor_storages,
                                           legacy_storage_map,
                                           error)) {
            return false;
        }

        size_t offset_after_state_dict = state_dict_offset + state_dict_size;
        size_t storage_keys_size       = 0;
        if (!skip_pickle_object(buffer.data() + offset_after_state_dict,
                                buffer.size() - offset_after_state_dict,
                                &storage_keys_size)) {
            return false;
        }

        *storage_data_offset = offset_after_state_dict + storage_keys_size;
        return finalize_tensor_offsets(*storage_data_offset, legacy_storage_map);
    };

    size_t object_size_1 = 0;
    size_t offset        = 0;

    if (skip_pickle_object(buffer.data(), buffer.size(), &object_size_1) &&
        pickle_object_is_torch_magic_number(buffer.data(), object_size_1)) {
        offset += object_size_1;

        size_t object_size_2 = 0;
        if (!skip_pickle_object(buffer.data() + offset, buffer.size() - offset, &object_size_2)) {
            set_error(error, torch_legacy_diagnostics(file_path, buffer));
            return false;
        }
        uint32_t protocol_version = 0;
        if (!parse_pickle_uint32_object(buffer.data() + offset, object_size_2, &protocol_version) || protocol_version != 1001) {
            set_error(error, torch_legacy_diagnostics(file_path, buffer));
            return false;
        }
        offset += object_size_2;

        size_t object_size_3 = 0;
        if (!skip_pickle_object(buffer.data() + offset, buffer.size() - offset, &object_size_3)) {
            set_error(error, torch_legacy_diagnostics(file_path, buffer));
            return false;
        }
        offset += object_size_3;

        size_t state_dict_size = 0;
        if (!skip_pickle_object(buffer.data() + offset, buffer.size() - offset, &state_dict_size)) {
            set_error(error, torch_legacy_diagnostics(file_path, buffer));
            return false;
        }

        size_t storage_data_offset = 0;
        if (parse_state_dict_at(offset, state_dict_size, &storage_data_offset)) {
            return true;
        }

        if (error != nullptr && error->empty()) {
            set_error(error, torch_legacy_diagnostics(file_path, buffer));
        }
        return false;
    }

    size_t state_dict_size = 0;
    if (skip_pickle_object(buffer.data(), buffer.size(), &state_dict_size)) {
        size_t storage_data_offset = 0;
        if (parse_state_dict_at(0, state_dict_size, &storage_data_offset)) {
            return true;
        }
    }

    if (error != nullptr && error->empty()) {
        set_error(error, torch_legacy_diagnostics(file_path, buffer));
    }
    return false;
}
