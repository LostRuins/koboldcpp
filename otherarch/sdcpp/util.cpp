#include "util.h"
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <locale>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include "preprocessing.hpp"

#include <inttypes.h>
#include <cinttypes>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "ggml-cpu.h"
#include "ggml.h"
#include "stable-diffusion.h"

bool ends_with(const std::string& str, const std::string& ending) {
    if (str.length() >= ending.length()) {
        return (str.compare(str.length() - ending.length(), ending.length(), ending) == 0);
    } else {
        return false;
    }
}

bool starts_with(const std::string& str, const std::string& start) {
    if (str.find(start) == 0) {
        return true;
    }
    return false;
}

bool contains(const std::string& str, const std::string& substr) {
    if (str.find(substr) != std::string::npos) {
        return true;
    }
    return false;
}

void replace_all_chars(std::string& str, char target, char replacement) {
    for (size_t i = 0; i < str.length(); ++i) {
        if (str[i] == target) {
            str[i] = replacement;
        }
    }
}

std::string sd_format(const char* fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(nullptr, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

int round_up_to(int value, int base) {
    if (base <= 0) {
        return value;
    }
    if (value % base == 0) {
        return value;
    } else {
        return ((value / base) + 1) * base;
    }
}

#ifdef _WIN32  // code for windows
#define NOMINMAX
#include <windows.h>

bool file_exists(const std::string& filename) {
    DWORD attributes = GetFileAttributesA(filename.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY));
}

bool is_directory(const std::string& path) {
    DWORD attributes = GetFileAttributesA(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY));
}

std::string sd_get_u8path(const std::string& file_path)
{
    return std::filesystem::u8path(file_path).string();
}

class MmapWrapperImpl : public MmapWrapper {
public:
    MmapWrapperImpl(void* data, size_t size, HANDLE hfile, HANDLE hmapping)
        : MmapWrapper(data, size), hfile_(hfile), hmapping_(hmapping) {}

    ~MmapWrapperImpl() override {
        UnmapViewOfFile(data_);
        CloseHandle(hmapping_);
        CloseHandle(hfile_);
    }

private:
    HANDLE hfile_;
    HANDLE hmapping_;
};

std::unique_ptr<MmapWrapper> MmapWrapper::create(const std::string& filename) {
    void* mapped_data = nullptr;
    size_t file_size  = 0;

    HANDLE file_handle = CreateFileA(
        filename.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (file_handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(file_handle, &size)) {
        CloseHandle(file_handle);
        return nullptr;
    }

    file_size = static_cast<size_t>(size.QuadPart);

    HANDLE mapping_handle = CreateFileMapping(file_handle, NULL, PAGE_READONLY, 0, 0, NULL);

    if (mapping_handle == NULL) {
        CloseHandle(file_handle);
        return nullptr;
    }

    mapped_data = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, file_size);

    if (mapped_data == NULL) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        return nullptr;
    }

    return std::make_unique<MmapWrapperImpl>(mapped_data, file_size, file_handle, mapping_handle);
}

#else  // Unix
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

bool is_directory(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

std::string sd_get_u8path(const std::string& file_path)
{
    return std::filesystem::path(file_path).string();
}

class MmapWrapperImpl : public MmapWrapper {
public:
    MmapWrapperImpl(void* data, size_t size)
        : MmapWrapper(data, size) {}

    ~MmapWrapperImpl() override {
        munmap(data_, size_);
    }
};

std::unique_ptr<MmapWrapper> MmapWrapper::create(const std::string& filename) {
    int file_descriptor = open(filename.c_str(), O_RDONLY);
    if (file_descriptor == -1) {
        return nullptr;
    }

    int mmap_flags = MAP_PRIVATE;

#ifdef __linux__
    // performance flags used by llama.cpp
    // posix_fadvise(file_descriptor, 0, 0, POSIX_FADV_SEQUENTIAL);
    // mmap_flags |= MAP_POPULATE;
#endif

    struct stat sb;
    if (fstat(file_descriptor, &sb) == -1) {
        close(file_descriptor);
        return nullptr;
    }

    size_t file_size = sb.st_size;

    void* mapped_data = mmap(NULL, file_size, PROT_READ, mmap_flags, file_descriptor, 0);

    close(file_descriptor);

    if (mapped_data == MAP_FAILED) {
        return nullptr;
    }

#ifdef __linux__
    // performance flags used by llama.cpp
    // posix_madvise(mapped_data, file_size, POSIX_MADV_WILLNEED);
#endif

    return std::make_unique<MmapWrapperImpl>(mapped_data, file_size);
}

#endif

bool MmapWrapper::copy_data(void* buf, size_t n, size_t offset) const {
    if (offset >= size_ || n > (size_ - offset)) {
        return false;
    }
    std::memcpy(buf, data() + offset, n);
    return true;
}

// get_num_physical_cores is copy from
// https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
// LICENSE: https://github.com/ggerganov/llama.cpp/blob/master/LICENSE
int32_t sd_get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break;  // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (siblings.size() > 0) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, nullptr, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, nullptr, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32)
    // TODO: Implement
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

static sd_progress_cb_t sd_progress_cb = nullptr;
void* sd_progress_cb_data              = nullptr;

static sd_preview_cb_t sd_preview_cb = nullptr;
static void* sd_preview_cb_data      = nullptr;
preview_t sd_preview_mode            = PREVIEW_NONE;
int sd_preview_interval              = 1;
bool sd_preview_denoised             = true;
bool sd_preview_noisy                = false;

std::u32string utf8_to_utf32(const std::string& utf8_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

std::string utf32_to_utf8(const std::u32string& utf32_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

std::u32string unicode_value_to_utf32(int unicode_value) {
    std::u32string utf32_string = {static_cast<char32_t>(unicode_value)};
    return utf32_string;
}

static std::string sd_basename(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    pos = path.find_last_of('\\');
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

std::string path_join(const std::string& p1, const std::string& p2) {
    if (p1.empty()) {
        return p2;
    }

    if (p2.empty()) {
        return p1;
    }

    if (p1[p1.length() - 1] == '/' || p1[p1.length() - 1] == '\\') {
        return p1 + p2;
    }

    return p1 + "/" + p2;
}

std::vector<std::string> split_string(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end   = str.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + 1;
        end   = str.find(delimiter, start);
    }

    // Add the last segment after the last delimiter
    result.push_back(str.substr(start));

    return result;
}

// { kcpp
static int sdloglevel = 0; //-1 = hide all, 0 = normal, 1 = showall
static bool sdquiet = false;
// } kcpp

static std::string build_progress_bar(int step, int steps) {
    std::string progress = "  |";
    int max_progress     = 50;
    int32_t current      = 0;
    if (steps > 0) {
        current = (int32_t)(step * 1.f * max_progress / steps);
    }
    for (int i = 0; i < 50; i++) {
        if (i > current) {
            progress += " ";
        } else if (i == current && i != max_progress - 1) {
            progress += ">";
        } else {
            progress += "=";
        }
    }
    progress += "|";
    return progress;
}

static void print_progress_line(int step, int steps, const std::string& speed_text) {
    if (step == 0) {
        return;
    }
    std::string progress = build_progress_bar(step, steps);
    const char* lf       = (step == steps ? "\n" : "");
    printf("\r%s %i/%i - %s\033[K%s", progress.c_str(), step, steps, speed_text.c_str(), lf);
    fflush(stdout);  // for linux
}
void pretty_progress(int step, int steps, float time) {
    if (sd_progress_cb) {
        sd_progress_cb(step, steps, time, sd_progress_cb_data);
        return;
    }
    if (step == 0) {
        return;
    }
    if(sdloglevel<0 || sdquiet)
    {
        return;
    }
    const char* unit = "s/it";
    float speed      = time;
    if (speed < 1.0f && speed > 0.f) {
        speed = 1.0f / speed;
        unit  = "it/s";
    }
    print_progress_line(step, steps, sd_format("%.2f%s", speed, unit));
}

void pretty_bytes_progress(int step, int steps, uint64_t bytes_processed, float elapsed_seconds) {
    if (sd_progress_cb) {
        float time = elapsed_seconds / (step + 1e-6f);
        sd_progress_cb(step, steps, time, sd_progress_cb_data);
        return;
    }
    if (step == 0) {
        return;
    }

    double bytes_per_second = 0.0;
    if (elapsed_seconds > 0.0f) {
        bytes_per_second = bytes_processed / (double)elapsed_seconds;
    }

    double speed_mb = bytes_per_second / (1024.0 * 1024.0);
    if (speed_mb >= 1024.0) {
        print_progress_line(step, steps, sd_format("%.2fGB/s", speed_mb / 1024.0));
    } else {
        print_progress_line(step, steps, sd_format("%.2fMB/s", speed_mb));
    }
}

std::string ltrim(const std::string& s) {
    auto it = std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    });
    return std::string(it, s.end());
}

std::string rtrim(const std::string& s) {
    auto it = std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    });
    return std::string(s.begin(), it.base());
}

std::string trim(const std::string& s) {
    return rtrim(ltrim(s));
}

static sd_log_cb_t sd_log_cb = nullptr;
void* sd_log_cb_data         = nullptr;

#define LOG_BUFFER_SIZE 4096

void log_message(const char* format, ...) {
    if (sdloglevel>0) {
        printf("\n");
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
        fflush(stdout);
    }
}
void set_sd_log_level(int log)
{
    sdloglevel = log;
}
bool get_sd_log_level()
{
    return sdloglevel;
}
void set_sd_quiet(bool quiet)
{
    sdquiet = quiet;
}

void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);

    static char log_buffer[LOG_BUFFER_SIZE + 1];
    int written = snprintf(log_buffer, LOG_BUFFER_SIZE, "%s:%-4d - ", sd_basename(file).c_str(), line);

    if (written >= 0 && written < LOG_BUFFER_SIZE) {
        vsnprintf(log_buffer + written, LOG_BUFFER_SIZE - written, format, args);
    }
    size_t len = strlen(log_buffer);
    if (log_buffer[len - 1] != '\n') {
        strncat(log_buffer, "\n", LOG_BUFFER_SIZE - len);
    }

    if (sd_log_cb) {
        sd_log_cb(level, log_buffer, sd_log_cb_data);
    }

    va_end(args);
}

void sd_set_log_callback(sd_log_cb_t cb, void* data) {
    sd_log_cb      = cb;
    sd_log_cb_data = data;
}
void sd_set_progress_callback(sd_progress_cb_t cb, void* data) {
    sd_progress_cb      = cb;
    sd_progress_cb_data = data;
}
void sd_set_preview_callback(sd_preview_cb_t cb, preview_t mode, int interval, bool denoised, bool noisy, void* data) {
    sd_preview_cb       = cb;
    sd_preview_cb_data  = data;
    sd_preview_mode     = mode;
    sd_preview_interval = interval;
    sd_preview_denoised = denoised;
    sd_preview_noisy    = noisy;
}

sd_preview_cb_t sd_get_preview_callback() {
    return sd_preview_cb;
}
void* sd_get_preview_callback_data() {
    return sd_preview_cb_data;
}

preview_t sd_get_preview_mode() {
    return sd_preview_mode;
}
int sd_get_preview_interval() {
    return sd_preview_interval;
}
bool sd_should_preview_denoised() {
    return sd_preview_denoised;
}
bool sd_should_preview_noisy() {
    return sd_preview_noisy;
}

sd_progress_cb_t sd_get_progress_callback() {
    return sd_progress_cb;
}
void* sd_get_progress_callback_data() {
    return sd_progress_cb_data;
}
const char* sd_get_system_info() {
    static char buffer[1024];
    std::stringstream ss;
    ss << "System Info: \n";
    ss << "    SSE3 = " << ggml_cpu_has_sse3() << " | ";
    ss << "    AVX = " << ggml_cpu_has_avx() << " | ";
    ss << "    AVX2 = " << ggml_cpu_has_avx2() << " | ";
    ss << "    AVX512 = " << ggml_cpu_has_avx512() << " | ";
    ss << "    AVX512_VBMI = " << ggml_cpu_has_avx512_vbmi() << " | ";
    ss << "    AVX512_VNNI = " << ggml_cpu_has_avx512_vnni() << " | ";
    ss << "    FMA = " << ggml_cpu_has_fma() << " | ";
    ss << "    NEON = " << ggml_cpu_has_neon() << " | ";
    ss << "    ARM_FMA = " << ggml_cpu_has_arm_fma() << " | ";
    ss << "    F16C = " << ggml_cpu_has_f16c() << " | ";
    ss << "    FP16_VA = " << ggml_cpu_has_fp16_va() << " | ";
    ss << "    WASM_SIMD = " << ggml_cpu_has_wasm_simd() << " | ";
    ss << "    VSX = " << ggml_cpu_has_vsx() << " | ";
    snprintf(buffer, sizeof(buffer), "%s", ss.str().c_str());
    return buffer;
}

sd_image_t tensor_to_sd_image(const sd::Tensor<float>& tensor, int frame_index) {
    const auto& shape = tensor.shape();
    GGML_ASSERT(shape.size() == 4 || shape.size() == 5);
    int width     = static_cast<int>(shape[0]);
    int height    = static_cast<int>(shape[1]);
    int channel   = static_cast<int>(shape[shape.size() == 5 ? 3 : 2]);
    uint8_t* data = (uint8_t*)malloc(static_cast<size_t>(width * height * channel));
    GGML_ASSERT(data != nullptr);

    for (int iw = 0; iw < width; ++iw) {
        for (int ih = 0; ih < height; ++ih) {
            for (int ic = 0; ic < channel; ++ic) {
                float value                            = shape.size() == 5 ? tensor.index(iw, ih, frame_index, ic, 0)
                                                                           : tensor.index(iw, ih, ic, frame_index);
                value                                  = std::clamp(value, 0.0f, 1.0f);
                data[(ih * width + iw) * channel + ic] = static_cast<uint8_t>(std::round(value * 255.0f));
            }
        }
    }
    return {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        static_cast<uint32_t>(channel),
        data,
    };
}

sd::Tensor<float> sd_image_to_tensor(sd_image_t image,
                                     int target_width,
                                     int target_height,
                                     bool scale) {
    sd::Tensor<float> tensor = sd::zeros<float>({static_cast<int64_t>(image.width),
                                                 static_cast<int64_t>(image.height),
                                                 static_cast<int64_t>(image.channel),
                                                 1});
    for (uint32_t iw = 0; iw < image.width; ++iw) {
        for (uint32_t ih = 0; ih < image.height; ++ih) {
            for (uint32_t ic = 0; ic < image.channel; ++ic) {
                tensor.index(iw, ih, ic, 0) = sd_image_get_f32(image, iw, ih, ic, scale);
            }
        }
    }
    if (target_width >= 0 && target_height >= 0 &&
        (tensor.shape()[0] != target_width || tensor.shape()[1] != target_height)) {
        tensor = sd::ops::interpolate(tensor,
                                      {target_width,
                                       target_height,
                                       tensor.shape()[2],
                                       tensor.shape()[3]});
    }
    return tensor;
}

// Constants for means and std
float means[3] = {0.48145466f, 0.4578275f, 0.40821073f};
float stds[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

sd::Tensor<float> clip_preprocess(const sd::Tensor<float>& image, int target_width, int target_height) {
    GGML_ASSERT(image.dim() == 4);
    GGML_ASSERT(image.shape()[2] == 3);
    GGML_ASSERT(image.shape()[3] == 1);
    GGML_ASSERT(target_width > 0 && target_height > 0);

    float width_scale  = static_cast<float>(target_width) / static_cast<float>(image.shape()[0]);
    float height_scale = static_cast<float>(target_height) / static_cast<float>(image.shape()[1]);
    float scale        = std::fmax(width_scale, height_scale);

    int64_t resized_width  = static_cast<int64_t>(scale * static_cast<float>(image.shape()[0]));
    int64_t resized_height = static_cast<int64_t>(scale * static_cast<float>(image.shape()[1]));

    sd::Tensor<float> resized = sd::ops::interpolate(
        image,
        {resized_width, resized_height, image.shape()[2], image.shape()[3]});

    int64_t h_offset = std::max<int64_t>((resized_height - target_height) / 2, 0);
    int64_t w_offset = std::max<int64_t>((resized_width - target_width) / 2, 0);

    sd::Tensor<float> cropped({target_width, target_height, image.shape()[2], image.shape()[3]});
    for (int64_t y = 0; y < target_height; ++y) {
        for (int64_t x = 0; x < target_width; ++x) {
            for (int64_t c = 0; c < image.shape()[2]; ++c) {
                cropped.index(x, y, c, 0) = resized.index(x + w_offset, y + h_offset, c, 0);
            }
        }
    }

    sd::Tensor<float> normalized = sd::ops::clamp(cropped, 0.0f, 1.0f);
    sd::Tensor<float> mean({1, 1, 3, 1}, {means[0], means[1], means[2]});
    sd::Tensor<float> std({1, 1, 3, 1}, {stds[0], stds[1], stds[2]});
    return (normalized - mean) / std;
}

// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/prompt_parser.py#L345
//
// Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
// Accepted tokens are:
//   (abc) - increases attention to abc by a multiplier of 1.1
//   (abc:3.12) - increases attention to abc by a multiplier of 3.12
//   [abc] - decreases attention to abc by a multiplier of 1.1
//   BREAK - separates the prompt into conceptually distinct parts for sequential processing
//   B - internal helper pattern; prevents 'B' in 'BREAK' from being consumed as normal text
//   \( - literal character '('
//   \[ - literal character '['
//   \) - literal character ')'
//   \] - literal character ']'
//   \\ - literal character '\'
//   anything else - just text
//
// >>> parse_prompt_attention('normal text')
// [['normal text', 1.0]]
// >>> parse_prompt_attention('an (important) word')
// [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
// >>> parse_prompt_attention('(unbalanced')
// [['unbalanced', 1.1]]
// >>> parse_prompt_attention('\(literal\]')
// [['(literal]', 1.0]]
// >>> parse_prompt_attention('(unnecessary)(parens)')
// [['unnecessaryparens', 1.1]]
// >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
// [['a ', 1.0],
//  ['house', 1.5730000000000004],
//  [' ', 1.1],
//  ['on', 1.0],
//  [' a ', 1.1],
//  ['hill', 0.55],
//  [', sun, ', 1.1],
//  ['sky', 1.4641000000000006],
//  ['.', 1.1]]
std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    float round_bracket_multiplier  = 1.1f;
    float square_bracket_multiplier = 1 / 1.1f;

    std::regex re_attention(R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|\bBREAK\b|[^\\()\[\]:B]+|:|\bB)");
    std::regex re_break(R"(\s*\bBREAK\b\s*)");

    auto multiply_range = [&](int start_position, float multiplier) {
        for (int p = start_position; p < res.size(); ++p) {
            res[p].second *= multiplier;
        }
    };

    std::smatch m, m2;
    std::string remaining_text = text;

    while (std::regex_search(remaining_text, m, re_attention)) {
        std::string text   = m[0];
        std::string weight = m[1];

        if (text == "(") {
            round_brackets.push_back((int)res.size());
        } else if (text == "[") {
            square_brackets.push_back((int)res.size());
        } else if (!weight.empty()) {
            if (!round_brackets.empty()) {
                multiply_range(round_brackets.back(), std::stof(weight));
                round_brackets.pop_back();
            }
        } else if (text == ")" && !round_brackets.empty()) {
            multiply_range(round_brackets.back(), round_bracket_multiplier);
            round_brackets.pop_back();
        } else if (text == "]" && !square_brackets.empty()) {
            multiply_range(square_brackets.back(), square_bracket_multiplier);
            square_brackets.pop_back();
        } else if (text == "\\(") {
            res.push_back({text.substr(1), 1.0f});
        } else if (std::regex_search(text, m2, re_break)) {
            res.push_back({"BREAK", -1.0f});
        } else {
            res.push_back({text, 1.0f});
        }

        remaining_text = m.suffix();
    }

    for (int pos : round_brackets) {
        multiply_range(pos, round_bracket_multiplier);
    }

    for (int pos : square_brackets) {
        multiply_range(pos, square_bracket_multiplier);
    }

    if (res.empty()) {
        res.push_back({"", 1.0f});
    }

    int i = 0;
    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + i + 1);
        } else {
            ++i;
        }
    }

    return res;
}
