#include "torch_zip_io.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "pickle_io.h"

#include "zip.h"

static void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

bool is_torch_zip_file(const std::string& file_path) {
    zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == nullptr) {
        return false;
    }
    zip_close(zip);
    return true;
}

static bool find_zip_entry(zip_t* zip, const std::string& entry_name, int* index, uint64_t* size) {
    size_t n = zip_entries_total(zip);
    for (size_t i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        std::string name = zip_entry_name(zip);
        if (name == entry_name) {
            *index = (int)i;
            *size  = zip_entry_size(zip);
            zip_entry_close(zip);
            return true;
        }
        zip_entry_close(zip);
    }
    return false;
}

static bool parse_zip_data_pkl(const uint8_t* buffer,
                               size_t buffer_size,
                               zip_t* zip,
                               const std::string& dir,
                               std::vector<TensorStorage>& tensor_storages,
                               std::string* error) {
    std::vector<TensorStorage> parsed_tensors;
    std::unordered_map<std::string, uint64_t> storage_nbytes;
    if (!parse_torch_state_dict_pickle(buffer, buffer_size, parsed_tensors, storage_nbytes, error)) {
        if (error != nullptr && error->empty()) {
            *error = "failed to parse torch zip pickle metadata";
        }
        return false;
    }

    for (auto& tensor_storage : parsed_tensors) {
        if (tensor_storage.storage_key.empty()) {
            set_error(error, "tensor '" + tensor_storage.name + "' has no storage key");
            return false;
        }

        const std::string entry_name = dir + "data/" + tensor_storage.storage_key;
        int zip_index                = -1;
        uint64_t entry_size          = 0;
        if (!find_zip_entry(zip, entry_name, &zip_index, &entry_size)) {
            set_error(error, "storage entry '" + entry_name + "' was not found");
            return false;
        }

        auto it_storage_size = storage_nbytes.find(tensor_storage.storage_key);
        if (it_storage_size != storage_nbytes.end() && entry_size < it_storage_size->second) {
            set_error(error, "storage entry '" + entry_name + "' is smaller than pickle metadata");
            return false;
        }

        uint64_t tensor_nbytes = tensor_storage.nbytes_to_read();
        if (tensor_storage.offset + tensor_nbytes > entry_size) {
            set_error(error, "tensor '" + tensor_storage.name + "' exceeds storage entry '" + entry_name + "'");
            return false;
        }

        tensor_storage.index_in_zip = zip_index;
        tensor_storage.storage_key.clear();
        tensor_storages.push_back(tensor_storage);
    }

    return true;
}

bool read_torch_zip_file(const std::string& file_path,
                         std::vector<TensorStorage>& tensor_storages,
                         std::string* error) {
    zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == nullptr) {
        set_error(error, "failed to open '" + file_path + "'");
        return false;
    }

    tensor_storages.clear();
    bool success        = true;
    bool found_data_pkl = false;
    int n               = (int)zip_entries_total(zip);
    for (int i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        std::string name = zip_entry_name(zip);
        size_t pos       = name.find("data.pkl");
        if (pos != std::string::npos) {
            found_data_pkl  = true;
            std::string dir = name.substr(0, pos);
            void* pkl_data  = nullptr;
            size_t pkl_size = 0;
            zip_entry_read(zip, &pkl_data, &pkl_size);

            if (pkl_data == nullptr || pkl_size == 0) {
                set_error(error, "failed to read '" + name + "' from '" + file_path + "'");
                success = false;
            } else if (!parse_zip_data_pkl((const uint8_t*)pkl_data, pkl_size, zip, dir, tensor_storages, error)) {
                success = false;
            }

            free(pkl_data);
        }
        zip_entry_close(zip);

        if (!success) {
            break;
        }
    }

    if (success && !found_data_pkl) {
        set_error(error, "data.pkl was not found in '" + file_path + "'");
        success = false;
    }

    zip_close(zip);
    return success;
}
