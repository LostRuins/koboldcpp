#include "utils.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <locale>
#include <codecvt>
#include <sstream>
#include <ctime>

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
// #define MA_API static
#include "miniaudio/miniaudio.h"

void utreplace(std::string & str, const std::string & needle, const std::string & replacement) {
    size_t pos = 0;
    while ((pos = str.find(needle, pos)) != std::string::npos) {
        str.replace(pos, needle.length(), replacement);
        pos += replacement.length();
    }
}

std::map<std::string, int32_t> json_parse(const std::string & fname) {
    std::map<std::string, int32_t> result;

    // read file into string
    std::string json;
    {
        std::ifstream ifs(fname);
        if (!ifs) {
            fprintf(stderr, "Failed to open %s\n", fname.c_str());
            exit(1);
        }

        json = std::string((std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
    }

    if (json[0] != '{') {
        return result;
    }

    // parse json
    {
        bool has_key  = false;
        bool in_token = false;

        std::string str_key = "";
        std::string str_val = "";

        int n = json.size();
        for (int i = 1; i < n; ++i) {
            if (!in_token) {
                if (json[i] == ' ') continue;
                if (json[i] == '"') {
                    in_token = true;
                    continue;
                }
            } else {
                if (json[i] == '\\' && i+1 < n) {
                    if (has_key == false) {
                        str_key += json[i];
                    } else {
                        str_val += json[i];
                    }
                    ++i;
                } else if (json[i] == '"') {
                    if (has_key == false) {
                        has_key = true;
                        ++i;
                        while (json[i] == ' ') ++i;
                        ++i; // :
                        while (json[i] == ' ') ++i;
                        if (json[i] != '\"') {
                            while (json[i] != ',' && json[i] != '}') {
                                str_val += json[i++];
                            }
                            has_key = false;
                        } else {
                            in_token = true;
                            continue;
                        }
                    } else {
                        has_key = false;
                    }

                    ::utreplace(str_key, "\\u0120", " " ); // \u0120 -> space
                    ::utreplace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    ::utreplace(str_key, "\\\"",    "\""); // \\\"   -> "

                    try {
                        result[str_key] = std::stoi(str_val);
                    } catch (...) {
                        //fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(), str_key.c_str(), str_val.c_str());

                    }
                    str_key = "";
                    str_val = "";
                    in_token = false;
                    continue;
                }
                if (has_key == false) {
                    str_key += json[i];
                } else {
                    str_val += json[i];
                }
            }
        }
    }

    return result;
}


void gpt_vocab::add_special_token(const std::string & token) {
    special_tokens.push_back(token);
}


std::string convert_to_utf8(const std::wstring & input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(input);
}


std::wstring convert_to_wstring(const std::string & input) {
    try {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(input);
    } catch (const std::range_error& e) {
        return L"";
    } catch (...) {
        return L"";
    }
}

void gpt_split_words(std::string str, std::vector<std::string>& words) {
    const std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::regex re(pattern);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;

        // Generate the subpattern from the special_tokens vector if it's not empty
        if (!vocab.special_tokens.empty()) {
            const std::regex escape(R"([\[\\\^\$\.\|\?\*\+\(\)\{\}])");
            std::string special_tokens_subpattern;
            for (const auto & token : vocab.special_tokens) {
                if (!special_tokens_subpattern.empty()) {
                    special_tokens_subpattern += "|";
                }
                special_tokens_subpattern += std::regex_replace(token, escape, R"(\$&)");
            }

            std::regex re(special_tokens_subpattern);
            std::smatch m;
            // Split the text by special tokens.
            while (std::regex_search(str, m, re)) {
                // Split the substrings in-between special tokens into words.
                gpt_split_words(m.prefix(), words);
                // Add matched special tokens as words.
                for (auto x : m) {
                    words.push_back(x);
                }
                str = m.suffix();
            }
            // Remaining text without special tokens will be handled below.
        }

        gpt_split_words(str, words);
    }

    // find the longest token that forms each word in words:
    std::vector<gpt_vocab::id> tokens;
    for (const auto & word : words) {
        for (int i = 0; i < word.size(); ){
            for (int j = word.size() - 1; j >= i; j--){
                auto cand = word.substr(i, j-i+1);
                auto it = vocab.token_to_id.find(cand);
                if (it != vocab.token_to_id.end()){ // word.substr(i, j-i+1) in vocab
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                }
                else if (j == i){ // word.substr(i, 1) has no matching
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                    i++;
                }
            }
        }
    }


    return tokens;
}

bool should_transpose_layer(std::string name)
{

    if(name.find(".mlp.fc_in.weight")!=std::string::npos ||
    name.find(".attn.out_proj.weight")!=std::string::npos ||
    name.find(".attn.q_proj.weight")!=std::string::npos ||
    name.find(".attn.k_proj.weight")!=std::string::npos ||
    name.find(".attn.v_proj.weight")!=std::string::npos ||
    name.find("/attn/c_attn/w")!=std::string::npos ||
    name.find("/attn/c_proj/w")!=std::string::npos ||
    name.find("/mlp/c_fc/w")!=std::string::npos ||
    name.find("/mlp/c_proj/w")!=std::string::npos)
    {
        return true;
    }
    return false;
}

static const std::string kcpp_base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";
static inline bool kcpp_is_base64(uint8_t c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}
std::vector<uint8_t> kcpp_base64_decode(const std::string & encoded_string)
{
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && kcpp_is_base64(encoded_string[in_]))
    {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4)
        {
            for (i = 0; i <4; i++)
            {
                char_array_4[i] = kcpp_base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++)
            {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j <4; j++)
        {
            char_array_4[j] = 0;
        }

        for (j = 0; j <4; j++)
        {
            char_array_4[j] = kcpp_base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; (j < i - 1); j++)
        {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}
std::string kcpp_base64_encode(const unsigned char* data, unsigned int data_length) {
    const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((data_length + 2) / 3) * 4);
    for (unsigned int i = 0; i < data_length; i += 3) {
        unsigned int triple = (data[i] << 16) + (i + 1 < data_length ? data[i + 1] << 8 : 0) + (i + 2 < data_length ? data[i + 2] : 0);
        encoded.push_back(base64_chars[(triple >> 18) & 0x3F]);
        encoded.push_back(base64_chars[(triple >> 12) & 0x3F]);
        if (i + 1 < data_length) {
            encoded.push_back(base64_chars[(triple >> 6) & 0x3F]);
        } else {
            encoded.push_back('=');
        }
        if (i + 2 < data_length) {
            encoded.push_back(base64_chars[triple & 0x3F]);
        } else {
            encoded.push_back('=');
        }
    }
    return encoded;
}
std::string kcpp_base64_encode(const std::string &data) {
    static const char lookup[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    int val = 0, valb = -6;
    for (unsigned char c : data) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(lookup[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        encoded.push_back(lookup[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (encoded.size() % 4) {
        encoded.push_back('=');
    }
    return encoded;
}

std::string get_timestamp_str()
{
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);
    char buffer[16]; // Buffer to hold "hh:mm:ss" and null terminator
    std::sprintf(buffer, "%02d:%02d:%02d", now->tm_hour, now->tm_min, now->tm_sec);
    // Convert the buffer to a std::string
    std::string timestamp(buffer);
    return timestamp;
}

//split a big vector into multiple small vectors of chunk size or less
std::vector<std::vector<int>> split_big_vector(const std::vector<int>& big_arr, size_t chunk_size) {
    std::vector<std::vector<int>> small_arrs;
    for (size_t i = 0; i < big_arr.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, big_arr.size());
        small_arrs.emplace_back(big_arr.begin() + i, big_arr.begin() + end);
    }
    return small_arrs;
}

std::vector<float> resample_wav(const std::vector<float>& input, uint32_t input_rate, uint32_t output_rate) {

    size_t input_size = input.size();

    double ratio = static_cast<double>(output_rate) / input_rate;
    size_t newLength = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(newLength);

    // Perform simple linear interpolation resampling
    for (size_t i = 0; i < newLength; ++i) {
        double srcIndex = i / ratio;
        size_t srcIndexInt = static_cast<size_t>(srcIndex);
        double frac = srcIndex - srcIndexInt;
        if (srcIndexInt + 1 < input_size) {
            output[i] = static_cast<float>(input[srcIndexInt] * (1 - frac) + input[srcIndexInt + 1] * frac);
        } else {
            output[i] = input[srcIndexInt];
        }
    }

    return output;
}

//a very rudimentary all in one sampling function which has no dependencies
int32_t kcpp_quick_sample(float * logits, const int n_logits, const std::vector<int32_t> & last_n_tokens, float rep_pen, float top_p, int top_k, float temp, std::mt19937 & rng)
{
    if (temp <= 0) {
        // select the token with the highest logit directly
        float max_logit = logits[0];
        int32_t max_id = 0;
        for (int i = 1; i < n_logits; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_id = i;
            }
        }
        return max_id;
    }

    top_k = (top_k<=0 || top_k>300)?300:top_k;
    top_k = std::min(top_k, n_logits);

    std::vector<std::pair<float, int32_t>> logits_id;
    logits_id.reserve(n_logits);

    //temperature sample
    const float scale = 1.0f/temp;

    //sample rep pen
    for (int i = 0; i < n_logits; ++i) {
        if (rep_pen>1.0f && std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
            // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if (logits[i] < 0.0f) {
                logits_id.push_back(std::make_pair((logits[i]*scale)*rep_pen, i));
            } else {
                logits_id.push_back(std::make_pair((logits[i]*scale)/rep_pen, i));
            }
        } else {
            logits_id.push_back(std::make_pair(logits[i]*scale, i));
        }
    }

    //sample top_k
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
        return a.first > b.first;
    });
    logits_id.resize(top_k);

    // compute probs for the top k tokens
    std::vector<float> probs;
    probs.reserve(logits_id.size());
    float maxl = logits_id[0].first;
    double sum = 0.0;
    for (const auto & kv : logits_id) {
        const float p = expf(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    //apply top p
    if (top_p < 1.0) {
        double cumsum = 0.0;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

void kcpp_embd_batch::init_kcpp_batch(int32_t n_tokens,
                                      int32_t npast,
                                      bool    use_mrope,
                                      bool    return_all_logits,
                                      bool    mrope_is_image,
                                      int     img_nx,
                                      int     img_ny) {
    const int          n_pos_per_embd = use_mrope ? 4 : 1;
    const llama_seq_id seq_id         = 0;

    if (use_mrope && mrope_is_image) {
        GGML_ASSERT(img_nx > 0 && img_ny > 0);
        GGML_ASSERT(img_nx * img_ny == n_tokens);
    }

    pos.resize(n_tokens * n_pos_per_embd);
    std::fill(pos.begin(), pos.end(), 0);

    n_seq_id.resize(n_tokens);
    seq_ids.resize(n_tokens + 1);
    logits.resize(n_tokens);
    seq_id_0.resize(1);

    seq_id_0[0]       = seq_id;
    seq_ids[n_tokens] = nullptr;

    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_ids.data();
    batch.logits   = logits.data();

    for (int i = 0; i < n_tokens; ++i) {
        n_seq_id[i] = 1;
        seq_ids[i]  = seq_id_0.data();
        logits[i]   = return_all_logits;
    }

    // ---- position encoding ----
    if (!use_mrope) {
        for (int i = 0; i < n_tokens; ++i) {
            pos[i] = npast + i;
        }
    } else if (!mrope_is_image) {
        // 1D M-RoPE (audio / embedding stream)
        for (int i = 0; i < n_tokens; ++i) {
            pos[i + 0 * n_tokens] = npast + i;
            pos[i + 1 * n_tokens] = npast + i;
            pos[i + 2 * n_tokens] = npast + i;
            pos[i + 3 * n_tokens] = 0;
        }
    } else {
        // 2D image M-RoPE
        int idx = 0;
        for (int y = 0; y < img_ny; ++y) {
            for (int x = 0; x < img_nx; ++x) {
                pos[idx + 0 * n_tokens] = npast;
                pos[idx + 1 * n_tokens] = npast + y;
                pos[idx + 2 * n_tokens] = npast + x;
                pos[idx + 3 * n_tokens] = 0;
                ++idx;
            }
        }
    }

    // Always request logits for last token
    logits[n_tokens - 1] = true;
}

//for embeddings
kcpp_embd_batch::kcpp_embd_batch(float * embd,
                                 int32_t n_tokens,
                                 int32_t npast,
                                 bool    use_mrope,
                                 bool    mrope_is_image,
                                 int     img_nx,
                                 int     img_ny) {
    batch = {
        /* n_tokens = */ n_tokens,
        /* tokens   = */ nullptr,
        /* embd     = */ embd,
        /* pos      = */ nullptr,
        /* n_seq_id = */ nullptr,
        /* seq_id   = */ nullptr,
        /* logits   = */ nullptr,
    };

    init_kcpp_batch(n_tokens, npast, use_mrope,
                    /*return_all_logits=*/false, mrope_is_image, img_nx, img_ny);
}

// for tokens
kcpp_embd_batch::kcpp_embd_batch(std::vector<llama_token> & tokens,
                                 int32_t                    npast,
                                 bool                       use_mrope,
                                 bool                       return_all_logits,
                                 bool                       mrope_is_image,
                                 int                        img_nx,
                                 int                        img_ny) {
    batch = {
        /* n_tokens = */ (int32_t) tokens.size(),
        /* tokens   = */ tokens.data(),
        /* embd     = */ nullptr,
        /* pos      = */ nullptr,
        /* n_seq_id = */ nullptr,
        /* seq_id   = */ nullptr,
        /* logits   = */ nullptr,
    };

    init_kcpp_batch(batch.n_tokens, npast, use_mrope, return_all_logits, mrope_is_image, img_nx, img_ny);
}

llama_batch kcpp_embd_batch::get_view(int offset, int n_tokens, int n_embd_mmproj) {
    GGML_ASSERT(offset >= 0);
    GGML_ASSERT(n_tokens > 0);
    GGML_ASSERT(offset + n_tokens <= batch.n_tokens);

    const int total_tokens = batch.n_tokens;
    llama_pos * pos_ptr = nullptr;

    // Detect M-RoPE vs normal RoPE
    const bool is_mrope = (pos.size() > (size_t)total_tokens);

    pos_view.clear();

    if (is_mrope) {
        const int n_pos_per_embd = pos.size() / total_tokens;
        GGML_ASSERT(n_pos_per_embd == 4);

        // Layout:
        // src: [dim0_all_tokens][dim1_all_tokens][dim2_all_tokens][dim3_all_tokens]
        // dst: same layout, but only [offset : offset + n_tokens]
        pos_view.reserve(n_tokens * n_pos_per_embd);

        for (int dim = 0; dim < n_pos_per_embd; ++dim) {
            const llama_pos * src =
                pos.data() + dim * total_tokens + offset;

            pos_view.insert(
                pos_view.end(),
                src,
                src + n_tokens
            );
        }

        pos_ptr = pos_view.data();
    }
    else {
        // Normal RoPE: contiguous slice
        pos_ptr = pos.data() + offset;
    }

    return {
        /* n_tokens = */ n_tokens,
        /* tokens   = */ nullptr,
        /* embd     = */ batch.embd ? batch.embd + offset*n_embd_mmproj : nullptr,
        /* pos      = */ pos_ptr,
        /* n_seq_id = */ batch.n_seq_id + offset,
        /* seq_id   = */ batch.seq_id   + offset,
        /* logits   = */ batch.logits   + offset,
    };
}

std::vector<std::string> split_string(const std::string& input, const std::string& separator) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = input.find(separator);

    while (end != std::string::npos) {
        result.push_back(input.substr(start, end - start));
        start = end + separator.length();
        end = input.find(separator, start);
    }

    // Add the remaining part after the last separator
    result.push_back(input.substr(start));

    return result;
}


static bool buf_is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// returns true if the buffer is a valid audio file
bool kcpp_decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    if (!buf_is_audio_file((const char *)buf_in, len))
    {
        return false;
    }

    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    ma_decoder_uninit(&decoder);
    return true;
}

static std::vector<std::string> kcpp_string_split(const std::string & input, char separator)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(begin_pos, separator_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = separator_pos + 1;
        separator_pos = input.find(separator, begin_pos);
    }
    parts.emplace_back(input.substr(begin_pos, separator_pos - begin_pos));
    return parts;
}

//for llama.cpp style device overrides e.g. --device Vulkan0,Vulkan1
std::vector<ggml_backend_dev_t> kcpp_parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = kcpp_string_split(value, ',');
    if (dev_names.empty()) {
        printf("\nkcpp_parse_device_list error: no devices specified\n");
        return std::vector<ggml_backend_dev_t>();
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        return std::vector<ggml_backend_dev_t>();
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                printf("\nkcpp_parse_device_list error: invalid device: %s\n",device.c_str());
                return std::vector<ggml_backend_dev_t>();
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}