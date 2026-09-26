#ifndef __SD_TOKENIZERS_VOCAB_VOCAB_H__
#define __SD_TOKENIZERS_VOCAB_VOCAB_H__

#include <string>

std::string load_clip_merges();
std::string load_qwen2_merges();
std::string load_mistral_merges();
std::string load_mistral_vocab_json();
std::string load_t5_tokenizer_json();
std::string load_umt5_tokenizer_json();
std::string load_gemma_merges();
std::string load_gemma_vocab_json();

#endif  // __SD_TOKENIZERS_VOCAB_VOCAB_H__