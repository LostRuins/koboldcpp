#include "phonemizer.h"

#include <fstream>

#ifdef ESPEAK_INSTALL
/**
 * espeak_wrapper functions and assignments
 *
 * The espeak_wrapper is a singleton which wraps threaded calls to espeak-ng with a shared mutex
 */

// non-const static members must be initialized out of line
espeak_wrapper* espeak_wrapper::instance{nullptr};
std::mutex espeak_wrapper::mutex;

espeak_wrapper * espeak_wrapper::get_instance() {
	if (!instance) {
		instance = new espeak_wrapper;
	}
	return instance;
}

const espeak_VOICE ** espeak_wrapper::list_voices() {
	std::lock_guard<std::mutex> lock(mutex);
	return espeak_ListVoices(nullptr);
}

espeak_ERROR espeak_wrapper::set_voice(const char * voice_code) {
	std::lock_guard<std::mutex> lock(mutex);
	return espeak_SetVoiceByName(voice_code);
}

const char * espeak_wrapper::text_to_phonemes(const void ** textptr, int textmode, int phonememode) {
	std::lock_guard<std::mutex> lock(mutex);
	return espeak_TextToPhonemes(textptr, textmode, phonememode);
}

void espeak_wrapper::initialize(espeak_AUDIO_OUTPUT output, int buflength, const char * path, int options) {
	std::lock_guard<std::mutex> lock(mutex);
	if (!espeak_initialized) {
		espeak_initialized = true;
		espeak_Initialize(output, buflength, path, options);
	}
}
#endif

/**
 * Helper functions for string parsing
 */
const std::unordered_set<std::string> inline_combine_sets(const std::vector<std::unordered_set<std::string>> sets) {
	std::unordered_set<std::string> combined;
	for (auto set : sets) {
		combined.insert(set.begin(), set.end());
	}
	return combined;
}

std::string replace(std::string target, char to_replace, char replacement) {
	for (int i = 0; i < target.size(); i++) {
		if (target[i] == to_replace) {
			target[i] = replacement;
		}
	}
	return target;
}

std::string to_lower(std::string word) {
	std::transform(word.begin(), word.end(), word.begin(),
    	[](unsigned char c){ return std::tolower(c);
    });
    return word;
}

std::string to_upper(std::string word) {
	std::transform(word.begin(), word.end(), word.begin(),
    	[](unsigned char c){ return std::toupper(c);
    });
    return word;
}

std::string replace_accents(std::string word) {
	std::string new_word;
	for (int i = 0; i < word.size();) {
		int grab = 0;
		while(i+grab+1 < word.size() && (word[i+grab + 1] & 0b11000000) == 0b10000000) {
			++grab;
		}
		++grab;

		if (grab > 1) {
			std::string accent = word.substr(i, grab);
			if (ACCENTED_A.find(accent) != std::string::npos) {
				new_word.push_back('a');
			} else if (ACCENTED_C.find(accent) != std::string::npos) {
				new_word.push_back('c');
			} else if (ACCENTED_E.find(accent) != std::string::npos) {
				new_word.push_back('e');
			} else if (ACCENTED_I.find(accent) != std::string::npos) {
				new_word.push_back('i');
			} else if (ACCENTED_N.find(accent) != std::string::npos) {
				new_word.push_back('n');
			} else if (ACCENTED_O.find(accent) != std::string::npos) {
				new_word.push_back('o');
			} else if (ACCENTED_U.find(accent) != std::string::npos) {
				new_word.push_back('u');
			} else {
				// non accented charactes in a word string should really be possible but for the sake of keeping this function pure
				// just put the multibyte character back;
				new_word.append(accent);

			}
		} else {
			new_word.push_back(word[i]);
		}
		i += grab;
	}
	return new_word;
}

int upper_count(std::string word) {
	int count = 0;
	for (char letter : word) {
		if (isupper(letter)) {
			count += 1;
		}
	}
	return count;
}

bool is_all_upper(std::string word) {
	for (char letter : word) {
		if (!isupper(letter)) {
			return false;
		}
	}
	return true;
}

/*
 * Text condition checks
 */
bool is_roman_numeral(char letter) {
	return ROMAN_NUMERAL_CHARACTERS.find(letter) != std::string::npos;
}

bool can_be_roman_numeral(std::string word) {
	for (int i = 0; i < word.size(); i++) {
		if (!is_roman_numeral(word[i])) {
			return false;
		}
	}
	return true;
}

bool is_alphabetic(char letter) {
	return ALPHABET.find(letter) != std::string::npos;
}

bool is_numeric(char letter) {
	int val = (int) letter;
	return val >= 48 && val <= 57;
}


std::string parse_voice_code(std::string voice_code) {
#ifdef ESPEAK_INSTALL
	voice_code = to_lower(voice_code);
	const espeak_VOICE * primary_match = nullptr;
	const espeak_VOICE * secondary_match = nullptr;
	bool search_by_lc = voice_code.size() == 2;
	bool search_by_lfc = !search_by_lc && voice_code.size() == 3;
	bool search_by_id = !search_by_lfc && voice_code.find("/") != std::string::npos;
	// It is common for locale's to be '_' separated rather than '-' separated. Check for both.
	bool search_by_lcc = !search_by_id && (voice_code.find("-") != std::string::npos || voice_code.find("_") != std::string::npos);
	if (search_by_id || search_by_lcc) {
		voice_code = replace(voice_code, '_', '-');
	}
	const espeak_VOICE** espeak_voices = espeak_wrapper::get_instance()->list_voices();
	// ideally we'd use the espeak voice scores which order voices by preference, but they are only returned when a voice_spec is passed to the list api and
	// the voice spec isn't compatible with partials (e.g. country codes, language family code, etc)
	int i = 0;
	while (espeak_voices[i] != nullptr) {
		auto identifier_parts = split(espeak_voices[i]->identifier, "/");
		// it is possible to add languages to espeak-ng without following their identifier pattern, if we run into such a language just try to match against
		// the identifier and otherwise continue;
		if (identifier_parts.size() == 1) {
			if (voice_code == identifier_parts[0] || voice_code == espeak_voices[i]->name) {
				primary_match = espeak_voices[i];
			} else {
				continue;
			}
		}
		if (search_by_lc) {
			std::string language_part = identifier_parts[1];
			if (language_part == voice_code) {
				primary_match = espeak_voices[i];
				break; // if we have an exact match then we can exit
			} else if (has_prefix(language_part, voice_code) && (!primary_match || strlen(primary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
				// prefer the smaller codes as longer codes typically refer to more specific locales
				primary_match = espeak_voices[i] ;
			} else {
				auto subparts = split(language_part, "-");
				if (subparts.size() > 1 && to_lower(subparts[1]) == voice_code && (!secondary_match || strlen(secondary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
					// country codes are typically capitalized in espeak-ng
					secondary_match = espeak_voices[i];
				}
			}
		} else if (search_by_lfc) {
			// espeak-ng uses language family codes in their identifiers, but also uses ISO 639-3 language codes for some languages.
			// Since language codes are more specific attempt to match against the language code as the primary and match against the language family
			// code as the secondary.
			if (has_prefix(identifier_parts[1], voice_code) && (!primary_match || strlen(primary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
				primary_match = espeak_voices[i];
			} else if (identifier_parts[0] == voice_code && (!secondary_match || strlen(secondary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
				secondary_match = espeak_voices[i];
			}
		} else if (search_by_id && has_prefix(to_lower(espeak_voices[i]->identifier), voice_code) && (!primary_match || strlen(primary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
			primary_match = espeak_voices[i];
		} else if (search_by_lcc && has_prefix(to_lower(identifier_parts[1]), voice_code) && (!primary_match || strlen(primary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
			primary_match = espeak_voices[i];
		} else if (to_lower(espeak_voices[i]->name).find(voice_code) != std::string::npos && (!primary_match || strlen(primary_match->identifier) > strlen(espeak_voices[i]->identifier))) {
			primary_match = espeak_voices[i];
		}
		i++;
	}
	if (!primary_match && !secondary_match) {
		TTS_ABORT("Failed to match espeak voice code '%s' to known espeak voices.\n", voice_code.c_str());
	}
	if (!primary_match) {
		primary_match = secondary_match;
	}
	fprintf(stdout, "Passed Espeak Voice Code '%s' doesn't directly match any known Espeak Voice IDs. Nearest match with name '%s' and id '%s' will be used instead.\n", voice_code.c_str(), primary_match->name, primary_match->identifier);
	return std::string(primary_match->identifier);
#else
	TTS_ABORT("Attempted to list voices without espeak-ng installed.");
#endif
}

void update_voice(std::string voice_code) {
#ifdef ESPEAK_INSTALL
	espeak_ERROR e = espeak_wrapper::get_instance()->set_voice(voice_code.c_str());
   	if (e != EE_OK) {
   		voice_code = parse_voice_code(voice_code);
   		espeak_wrapper::get_instance()->set_voice(voice_code.c_str());
    }
#else
    TTS_ABORT("Attempted to set voice without espeak-ng installed.");
#endif
}


void conditions::reset_for_clause_end() {
	hyphenated = false;
	was_punctuated_acronym = false;
	beginning_of_clause = true;
	was_number = false;
}

void conditions::reset_for_space() {
	hyphenated = false;
	was_punctuated_acronym = false;
	was_word = false;
}

void conditions::update_for_word(std::string word, bool allow_for_upper_check) {
	if (allow_for_upper_check && !is_all_upper(word)) {
		was_all_capitalized = false;
	}
	was_word = true;
	beginning_of_clause = false;
	hyphenated = false;
	was_number = false;
}

std::string corpus::next(int count) {
	if (location == size || count == 0) {
		return "";
	}
	int final_loc = location;
	int grabbed = 0;
	while(grabbed < count && final_loc < size) {
		while(final_loc + 1 < size && (text[final_loc+1] & 0b11000000) == 0b10000000) {
			++final_loc;
		}
		++final_loc;
		++grabbed;
	}
	return std::string(text+location, text+final_loc);
}

std::string corpus::last(int count) {
	if (location == 0 || count == 0) {
		return "";
	}
	int final_loc = location - 1;
	int grabbed = 0;
	while(grabbed < count && final_loc > 0) {
		while((text[final_loc] & 0b11000000) == 0b10000000) {
			--final_loc;
		}
		++grabbed;
	}

	return std::string(text+final_loc, text+location-1);
}

std::string corpus::pop(int count) {
	std::string ret = next(count);
	location += ret.size();
	return ret;
}

std::string corpus::after(int aftr, int count) {
	size_t new_loc = location + aftr;
	if (new_loc >= size || count == 0) {
		return "";
	}
	int final_loc = new_loc;
	int grabbed = 0;
	while(grabbed < count && final_loc < size) {
		while(final_loc+1 < size && (text[final_loc+1] & 0b11000000) == 0b10000000) {
			++final_loc;
		}
		++final_loc;
		++grabbed;
	}
	return std::string(text+new_loc, text+final_loc);
}

std::string corpus::size_pop(size_t pop_size) {
	size_t tsize = std::min(pop_size, size - location);
	std::string ret = std::string(text+location, text+location+tsize);
	location += tsize;
	return ret;
}

std::string corpus::next_in(std::string val, bool* has_accent) {
	int n = 0;
	int running = 0;
	std::string nafter = next();
	while (nafter != "" && val.find(nafter) != std::string::npos) {
		if (has_accent && !(*has_accent) && COMMON_ACCENTED_CHARACTERS.find(nafter) != std::string::npos) {
			*has_accent = true;
		}
		++n;
		running += nafter.size();
		nafter = after(running);
	}
	return next(n);
}

std::string corpus::pop_in(std::string val) {
	int n = 0;
	size_t running = 0;
	std::string nafter = next();
	running += nafter.size();
	while (nafter != "" && val.find(nafter) != std::string::npos) {
		++n;
		nafter = after(running);
		running += nafter.size();
	}
	return pop(n);
}

std::string corpus::after_until(int aftr, std::string val) {
	int n = 0;
	std::string nafter = after(aftr);
	while (nafter != "" && val.find(nafter) != std::string::npos) {
		++n;
		nafter = after(n);
	}
	return after(aftr, n);
}

std::string phonemizer_rule::lookup_rule(std::vector<std::string> & keys, int index) {
	if (index >= keys.size()) {
		return value;
	}
	std::string found_key = keys[index];
	bool found_match = false;
	for (const auto& pair : rules) {
		if (pair.first == found_key) {
			found_match = true;
			break;
		} else if (pair.first[0] == '*' && has_suffix(found_key, pair.first.substr(1))) {
			found_match = true;
			found_key = pair.first;
			break;
		} else if (pair.first.back() == '*' && has_prefix(found_key, pair.first.substr(0, pair.first.size()-1))) {
			found_match = true;
			found_key = pair.first;
			break;
		}
	}
	if (found_match) {
		return rules.at(found_key)->lookup_rule(keys, index + 1);
	} else {
		return value;
	}
}

std::string word_phonemizer::lookup_rule(std::string word, std::string current, std::string before, std::string after) {
	if (rules.find(current) == rules.end()) {
		return "";
	}
	std::vector<std::string> lookup_keys = {before, after, word};
	return rules[current]->lookup_rule(lookup_keys, 0);
}

void word_phonemizer::add_rule(std::vector<std::string> keys, std::string phoneme) {
	phonemizer_rule * current_rule = nullptr;
	for (int i = 0; i < keys.size(); i++) {
		if (current_rule) {
			if (current_rule->rules.find(keys[i]) == current_rule->rules.end()) {
				phonemizer_rule * nrule = new phonemizer_rule;
				current_rule->rules[keys[i]] = nrule;
				current_rule = nrule;
			} else {
				current_rule = current_rule->rules.at(keys[i]);
			}
		} else {
			if (rules.find(keys[i]) == rules.end()) {
				current_rule = new phonemizer_rule;
				rules[keys[i]] = current_rule;
			} else {
				current_rule = rules.at(keys[i]);
			}
		}
	}
	if (current_rule) {
		current_rule->value = phoneme;
	}
}

std::string word_phonemizer::phonemize(std::string word) {
	std::vector<std::string> graphemes;
	word = to_lower(word);
	tokenizer->token_split(word, graphemes);
	std::string phoneme = "";
	for (int i = 0; i < graphemes.size(); i++) {
		std::string before = i > 0 ? graphemes[i-1] : "^";
		std::string after = i + 1 < graphemes.size() ? graphemes[i+1] : "$";
		std::string current = graphemes[i];
		phoneme += lookup_rule(word, current, before, after);
	}
	return phoneme;
}

std::string build_subthousand_phoneme(int value) {
	int hundreds = value / 100;
	std::string phoneme = hundreds > 0 ? NUMBER_PHONEMES[hundreds] + " " + HUNDRED_PHONEME : "";
	value = value % 100;
	if (value > 0 && value < 20) {
		phoneme += NUMBER_PHONEMES[value];
	} else if (value > 0) {
		phoneme += SUB_HUNDRED_NUMBERS[(value / 10) - 2];
		value = value % 10;
		if (value > 0) {
			phoneme += " " + NUMBER_PHONEMES[value];
		}
	}
	return phoneme;
}

std::string build_number_phoneme(long long int remainder) {
	std::string phoneme = "";
	bool started = false;
	if (remainder > TRILLION) {
		long long int trillions = (long long int) remainder / TRILLION;
		phoneme += build_subthousand_phoneme(trillions) + " " + TRILLION_PHONEME;
		remainder = (long long int) remainder % TRILLION;
		if (remainder > 0) {
			phoneme += ",";
		}
		started = true;
	}
	if (remainder > BILLION) {
		long long int billions = (long long int) remainder / BILLION;
		remainder = (long long int) remainder % BILLION;
		std::string billion_part =  build_subthousand_phoneme(billions) + " " + BILLION_PHONEME;
		if (!started) {
			phoneme += remainder > 0 ? billion_part + "," : billion_part;

		} else if (remainder == 0) {
			phoneme += " " + billion_part;
		} else {
			phoneme += " " + billion_part + ",";
		}
		started = true;
	}
	if (remainder > MILLION) {
		long long int millions = (long long int) remainder / MILLION;
		remainder = (long long int) remainder % MILLION;
		std::string million_part =  build_subthousand_phoneme(millions) + " " + MILLION_PHONEME;
		if (!started) {
			phoneme += remainder > 0 ? million_part + "," : million_part;
		} else if (remainder == 0) {
			phoneme += " " + million_part;
		} else {
			phoneme += " " + million_part + ",";
		}
		started = true;
	}
	if (remainder > 1000) {
		long long int thousands = (long long int) remainder / 1000;
		remainder = (long long int) remainder % 1000;
		std::string thousand_part =  build_subthousand_phoneme(thousands) + " " + THOUSAND_PHONEME;
		if (!started) {
			phoneme += remainder > 0 ? thousand_part + "," : thousand_part;
		} else if (remainder == 0) {
			phoneme += " " + thousand_part;
		} else {
			phoneme += " " + thousand_part + ",";
		}
		started = true;
	}
	if (remainder > 0) {
		if (started) {
			phoneme += " " + build_subthousand_phoneme(remainder);
		} else {
			phoneme += build_subthousand_phoneme(remainder);
		}
	}
	return phoneme;
}

bool dictionary_response::is_successful() {
	return code < 200;
}

bool dictionary_response::is_match(corpus* text, conditions* flags) {
	if (not_at_clause_end) {
		std::string chunk = text->next_in(NON_CLAUSE_WORD_CHARACTERS);
		std::string after = text->after(chunk.size());
		if (after == "!" || after == "." || after == "?") {
			return false;
		}
	}
	return text->next(after_match.size()) == after_match && (!expects_to_be_proceeded_by_number || flags->was_number) && (!not_at_clause_start || !flags->beginning_of_clause);
}

dictionary_response * phoneme_dictionary::lookup(corpus * text, std::string value, conditions* flags) {
	if (lookup_map.find(value) == lookup_map.end()) {
		return not_found_response;
	}
	std::vector<dictionary_response*> possibilities = lookup_map.at(value);
	for (auto possible : possibilities) {
		if (possible->code == SUCCESS_TOTAL || (possible->code == SUCCESS_PARTIAL && possible->is_match(text, flags))) {
			return possible;
		}
	}
	return phonetic_fallback_response;
}

bool phonemizer::handle_space(corpus* text, std::string* output, conditions* flags) {
	flags->reset_for_space();
	text->pop_in(" \n\f\t");
	if (output->back() !=  ' ') {
		output->append(" ");
	}
	return true;
}

void phonemizer::append_numeric_series(std::string series, std::string* output, conditions * flags) {
	if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
		output->append(" ");
	}
	for (int i = 0; i < series.size(); i++) {
		int numeral = series[i] - '0';
		output->append(NUMBER_PHONEMES[numeral]);
		if (i + 1 < series.size()) {
			output->append(" ");
		}
	}
	if (series.size() > 0) {
		flags->update_for_word(series);
		flags->was_number = true;
	}
}

bool phonemizer::handle_numeric_series(corpus* text, std::string* output, conditions* flags) {
	std::string series = text->pop_in(NUMBER_CHARACTERS);
	append_numeric_series(series, output, flags);
	return true;
}

bool phonemizer::handle_numeric(corpus* text, std::string* output, conditions* flags) {
	/*
	 * There are four recognized ways of separating large arabic numerals:
	 *   1. No breaks or seperations exception for the decimal (e.g. '32000.012' or '32000,012')
	 *   2. Space separated breaks between every three digits and comma separated decimals (e.g. '32 000,012')
	 *   3. Period separated breaks between every three digits and comma separated decimals (e.g. '32.000,012')
	 *   4. Comma separated breaks between every three digits and period separated decimals (e.g. '32,000.012')
	 *
	 * This implementation will support all three approaches up to the trillions, after which numbers will be represented as a series
	 * of distinct digits. Non conforming patterns, e.g. multiple commas, multiple periods, or multiple spaces that are not three
	 * digits apart, will not be treated as continuous numbers but rather separate numerical strings.
	 */
 	std::string number = text->next_in(COMPATIBLE_NUMERICS);
 	number = strip(number, ",. ");

	// For numerics, we don't necessarily want to stop reading from the corpus at periods, commas, or spaces.
	char large_number_separator = '\0';
	char decimal_separator = '\0';
	char last_break_char = '\0';
	bool invalid_format = false;
	int count_since_break = 0;
	std::string built = "";
	for (char & c : number) {
		if (is_numeric(c)) {
			built += c;
			count_since_break += 1;
		} else if (last_break_char =='\0') {
			if (count_since_break > 3) {
				decimal_separator = c;
			}
			last_break_char = c;
			built += c;
			count_since_break = 0;
		} else if (c != last_break_char) {
			if (c == ' ') {
				break;
			} else if (count_since_break == 3 && decimal_separator == '\0') {
				if (large_number_separator == '\0') {
					large_number_separator = last_break_char;
				}
				decimal_separator = c;
				built += c;
				count_since_break = 0;
				last_break_char = c;
			} else if (count_since_break != 3) {
				if (large_number_separator != '\0') {
					invalid_format = true;
				}
				break;
			} else {
				break;
			}
		} else if (c == last_break_char) {
			if (decimal_separator != '\0') {
				break;
			} else if (count_since_break != 3) {
				invalid_format = true;
				break;
			} else {
				large_number_separator = c;
				built += c;
				count_since_break = 0;
			}
		}
	}

	if (!invalid_format) {
		if (large_number_separator != '\0' && decimal_separator == '\0' && count_since_break != 3) {
			invalid_format = true;
		} else if (count_since_break == 3 && last_break_char != '\0' && decimal_separator == '\0' && large_number_separator == '\0') {
			large_number_separator = last_break_char;
		} else if (count_since_break != 3 && last_break_char != '\0' && decimal_separator == '\0' && large_number_separator == '\0') {
			decimal_separator = last_break_char;
		}
	}

	if (invalid_format) {
		return handle_numeric_series(text, output, flags);
	}

	if (large_number_separator != '\0') {
		built.erase(std::remove(built.begin(), built.end(), large_number_separator), built.end());
	}
	if (decimal_separator == ',') {
		replace(built, decimal_separator, '.');
	}
	long long int value = std::stoll(built);

	if (value >= LARGEST_PRONOUNCABLE_NUMBER) {
		return handle_numeric_series(text, output, flags);
	}

	text->size_pop(built.size());

	std::string noutput = build_number_phoneme(value);
	if (noutput.size() > 0) {
		if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
			output->append(" ");
		}
		output->append(noutput);
		flags->update_for_word(built);
		flags->was_number = true;
	}
	if (decimal_separator != '\0') {
		std::vector<std::string> parts = split(built, decimal_separator);
		if (parts[1].size() > 0) {
			output->append(" " + POINT_PHONEME + " ");
			append_numeric_series(parts[1], output, flags);
		}
	}
	return true;
}

bool phonemizer::is_acronym_like(corpus* text, std::string word, conditions* flags) {
	if (word.find(".") != std::string::npos) {
		for (std::string part : split(word, ".")) {
			if (part.size() == 0) {
				return false;
			}
			if (part.size() > 1) {
				if (part.size() > 2 || !(isupper(part[0]) && islower(part[1]))) {
					return false;
				}
			}
		}
		return true;
	} else if (word.size() < 4) {
		return small_english_words.find(to_lower(word)) == small_english_words.end();
	} else if (is_all_upper(word)) {
		if (flags->was_all_capitalized || is_all_upper(text->after_until(word.size()+1, " "))) {
			flags->was_all_capitalized = true;
			return false;
		}
		return true;
	} else if (!is_all_upper(word) && upper_count(word) > (int) word.length() / 2) {
		return true;
	}
	return false;
}

bool phonemizer::handle_roman_numeral(corpus* text, std::string* output, conditions * flags) {
	auto next = text->next();
	next = to_lower(next);
	int total = 0;
	int last_value = 0;
	std::string running = "";
	while (is_roman_numeral(next[0])) {
		bool found = false;
		for (int size = 4; size > 0; size--) {
			std::string chunk = text->after(running.size(), size);
			chunk = to_lower(chunk);
			if (ROMAN_NUMERALS.find(chunk) != ROMAN_NUMERALS.end()) {
				found = true;
				int found_value = ROMAN_NUMERALS.at(chunk);
				if (total == 0 || last_value > found_value) {
					total += found_value;
					last_value = found_value;
					running += chunk;
				} else {
					return false;
				}
			}
		}
		if (found) {
			next = text->after(running.size());
			to_lower(next);
			continue;
		}
		return false;
	}

	std::string noutput = build_number_phoneme(total);
	if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
		output->append(" ");
	}
	output->append(noutput);
	text->size_pop(running.size());
	flags->update_for_word(running, false);
	flags->was_number = true;

	return true;
}

bool phonemizer::handle_acronym(corpus* text, std::string word, std::string* output, conditions * flags) {
	std::string out = "";
	for (int i = 0; i < word.size(); i++) {
		try {
			if (word[i] == '.') {
				flags->was_punctuated_acronym = true;
				continue;
			}
			char letter = std::tolower(word[i]);
			out += LETTER_PHONEMES.at(letter);
		} catch (const std::out_of_range& e) {
			continue;
		}
	}
	text->size_pop(word.size());
	if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
		output->append(" ");
	}
	output->append(out);
	flags->update_for_word(word, false);
	return true;
}

bool phonemizer::handle_phonetic(corpus* text, std::string word, std::string* output, conditions* flags, size_t unaccented_size_difference) {
	if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
		output->append(" ");
	}
	output->append(phonetic_phonemizer->phonemize(word));
	text->size_pop(word.size()+unaccented_size_difference);
	flags->update_for_word(word);
	return true;
}

static std::unordered_map<std::string, std::string> kokoro_ipa_map;
void populate_kokoro_ipa_map(std::string executable_path)
{
	std::string line;
    auto filepath = executable_path + "embd_res/kokoro_ipa.embd";
    printf("\nReading Kokoro IPA from %s",filepath.c_str());
    std::ifstream myfile(filepath);
    if (myfile.is_open())
    {
        while (myfile.good())
        {
            getline(myfile, line);
			auto parts = split(line, ",");
			if(parts.size()==2)
			{
                kokoro_ipa_map[parts[0]] = parts[1];
            } else {
                printf("\nError reading line in Kokoro IPA!");
            }
        }
        myfile.close();
		printf("\nPopulated Kokoro IPA: %d entries\n", kokoro_ipa_map.size());
    }
    else
    {
		printf("\nUnable to open Kokoro IPA file\n");
    }
}
std::string found_word_to_ipa(std::string input)
{
	bool is_acronym = !input.empty() &&
        std::all_of(input.begin(), input.end(), [](unsigned char c) {
            return std::isupper(c);
        });

    if (is_acronym) {
        return ""; // Return empty for acronyms
    }

	// Convert input to lowercase
    std::transform(input.begin(), input.end(), input.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    auto it = kokoro_ipa_map.find(input);
    if (it != kokoro_ipa_map.end()) {
        return it->second; // found
    }
    return "";
}
bool phonemizer::process_word(corpus* text, std::string* output, std::string word, conditions* flags, bool has_accent) {
	dictionary_response* response;
	size_t unaccented_size_difference = 0;

	std::string foundstr = found_word_to_ipa(word);
	if(foundstr!="") //do not use if its part of a contracted word
	{
		std::string detected = text->next(word.size()+1);
		if(detected.back()!='\'')
		{
			output->append(foundstr);
			text->size_pop(word.size());
			return true;
		}
	}

	if (has_accent) {
		response = dict->lookup(text, word, flags);
		if (!response->is_successful()) {
			unaccented_size_difference = word.size();
			word = replace_accents(word);
			unaccented_size_difference -= word.size();
			response = dict->lookup(text, word, flags);
		}
	} else {
		response = dict->lookup(text, word, flags);
	}

	//printf("\nSUCCESS: %d, word:%s, result:%s\n",response->is_successful(),word.c_str(),response->value.c_str());

	if (response->is_successful()) {
		if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
			output->append(" ");
		}
		flags->update_for_word(word);
		if (response->code != SUCCESS_TOTAL) {
			word += response->after_match;
			output->append(response->value);
			text->size_pop(word.size()+unaccented_size_difference);
			return true;
		} else {
			output->append(response->value);
			text->size_pop(word.size()+unaccented_size_difference);
			return true;
		}
	}
	// else if (can_be_roman_numeral(word) && is_all_upper(word) && small_english_words.find(to_lower(word)) == small_english_words.end() && handle_roman_numeral(text, output, flags)) {
	// 	return true;
	// }
	else if (is_acronym_like(text, word, flags)) {
		return handle_acronym(text, word, output, flags);
	} else if (word.find(".") < word.length()) {
		bool part_has_accent = false;
		std::string word_part = text->next_in(ALPHABET+COMMON_ACCENTED_CHARACTERS, &part_has_accent);
		process_word(text, output, word_part, flags, part_has_accent);
		handle_punctuation(text, ".", output, flags);
		output->append(" ");
		flags->reset_for_space();
		return true;
	} else {
		return handle_phonetic(text, word, output, flags, unaccented_size_difference);
	}
	return true;
}

bool phonemizer::handle_word(corpus * text, std::string* output, conditions * flags) {
	bool has_accent = false;
	std::string word = text->next_in(WORD_CHARACTERS, &has_accent);
	while (word.size() > 0 && word.back() == '.') {
		word = word.substr(0,word.size()-1);
	}

	return process_word(text, output, word, flags, has_accent);
}

bool phonemizer::handle_replacement(corpus* text, std::string next, std::string* output, conditions * flags) {
	if (flags->was_word && output->back() != ' ' && !flags->hyphenated) {
		output->append(" ");
	}
	output->append(REPLACEABLE.at(next));
	flags->update_for_word(next);
	text->pop();
	return true;
}

bool phonemizer::handle_possession_plural(corpus* text, std::string* output, conditions * flags) {
	if (text->next(2) == "'s") {
		std::string last = text->last();
		if (VOWELS.find(to_lower(last)[0]) != std::string::npos) {
			output->append("z");
		} else if (last == "s" || last == "z") {
			output->append("ᵻz");
		} else if (is_alphabetic(last[0])) {
			output->append("s");
		} else {
			output->append("ˈɛs");
		}
		text->pop(2);
	} else {
		text->pop();
	}
	return true;
}

bool phonemizer::handle_contraction(corpus* text, std::string* output, conditions * flags) {
	text->pop();
	std::string next = text->next_in(ALPHABET);
	next = to_lower(next);
	try {
		output->append(CONTRACTION_PHONEMES.at(next));
	} catch (const std::out_of_range& e) {
		// in the situation that we cannt find a contraction then we just want to pop the ' character and continue
		// it could be the end of a single quote which is ignored by the espeak phonemizer.
		return true;
	}
	// make sure to pop the contraction.
	text->pop_in(ALPHABET);
	return true;
}

bool phonemizer::handle_punctuation(corpus* text, std::string next, std::string* output, conditions * flags) {
	std::string last = text->last();
	std::string after = text->after();
	if (next[0] == '.') {
		if (flags->was_punctuated_acronym) {
			// we finished an acronym
			flags->was_punctuated_acronym = false;
			output->append(next);
			text->pop();
			if (text->after(1, 2) == "'s") {
				return handle_possession_plural(text, output, flags);
			}
			return true;
		}
		std::string chunk = text->next_in(".");
		/*if (chunk.size() > 1) {
			flags->pre_pause += 4;
		}*/
		output->append(chunk);
		text->size_pop(chunk.size());
		return true;
	} else if (next == "'") {
		if (flags->was_word  && (after == "s" || !is_alphabetic(after[0]))) {
			return handle_possession_plural(text, output, flags);
		} else if (flags->was_word && (CONTRACTION_PHONEMES.find(after) != CONTRACTION_PHONEMES.end() || CONTRACTION_PHONEMES.find(text->after(next.size(), 2)) != CONTRACTION_PHONEMES.end())) {
			return handle_contraction(text, output, flags);
		} else {
			// could be the end or start of a quote
			text->pop();
			return true;
		}
	}  else if (next[0] == '-') {
		if (last == " " && after == " ") {
			//flags->pre_pause += 4;
			text->pop(2);
			flags->reset_for_space();
			return true;
		} else if (after[0] == '-') {
			//flags->pre_pause += 4;
			text->pop(2);
			output->append(" ");
			flags->reset_for_space();
			return true;
		} else if (!flags->beginning_of_clause && flags->was_word && is_alphabetic(after[0])) {
			flags->hyphenated = true;
			text->pop();
			return true;
		} else {
			// ignore it
			text->pop();
			return true;
		}
	}
	else if (CLAUSE_BREAKS.find(next) != std::string::npos) {
		output->append(next);
		flags->reset_for_clause_end();
		text->pop();
		return true;
	} else if (NOOP_BREAKS.find(next) != std::string::npos) {
		output->append(next);
		text->pop();
		return true;
	} else if (REPLACEABLE.find(next) != REPLACEABLE.end()) {
		return handle_replacement(text, next, output, flags);
	} else {
		// ignore it
		text->pop();
		return true;
	}
}

bool phonemizer::route(corpus * text, std::string* output, conditions * flags) {
	std::string next = text->next();
	if (next == "") {
		// we finished lexing the corpus
		return false;
	}
	if (SPACE_CHARACTERS.find(next) != std::string::npos) {
		return handle_space(text, output, flags);
	} else if (is_numeric(next[0])) {
		return handle_numeric(text, output, flags);
	} else if (is_alphabetic(next[0])) {
		return handle_word(text, output, flags);
	} else {
		return handle_punctuation(text, next, output, flags);
	}
}

#ifdef ESPEAK_INSTALL
std::string phonemizer::espeak_text_to_phonemes(const char * text) {
	int mode = phoneme_mode == IPA ? (0 << 8 | 0x02) : (0 << 8 | 0x01);
	const void ** txt_ptr = (const void**)&text;
	const char * resp = espeak_wrapper::get_instance()->text_to_phonemes(txt_ptr, espeakCHARS_UTF8, mode);
	return strip(std::string(resp));
}
#endif

std::string phonemizer::text_to_phonemes(const char * text, size_t size) {
	std::string output = "";
	if (mode == ESPEAK) {
#ifdef ESPEAK_INSTALL
		auto parts = split(text, STOPPING_TOKENS, true);
		std::string phonemes = "";
		for (int i = 0; i < parts.size(); i+=2) {
			phonemes += espeak_text_to_phonemes(parts[i].c_str());
			if (preserve_punctuation && i + 1 < parts.size()) {
				phonemes += parts[i+1];
			}
		}
		return phonemes;
#else
		TTS_ABORT("%s attempted to run in espeak mode without espeak installed. \n", __func__);
#endif
	} else {
		text_to_phonemes(text, size, &output);
	}
	return output;
}

std::string phonemizer::text_to_phonemes(std::string text) {
	return text_to_phonemes(text.c_str(), text.size());
}

void phonemizer::text_to_phonemes(const char * text, size_t size, std::string* output) {
	if (mode == ESPEAK) {
#ifdef ESPEAK_INSTALL
		TTS_ABORT("%s attempted to run in espeak mode with output already defined. \n", __func__);
#else
		TTS_ABORT("%s attempted to run in espeak mode without espeak installed. \n", __func__);
#endif
		return;
	}
	corpus * corpus_text = new corpus(text, size);
	conditions * flags = new conditions;
	bool running = true;
	while (running) {
		running = route(corpus_text, output, flags);
	}
	delete corpus_text;
	delete flags;
}

void phonemizer::text_to_phonemes(std::string text, std::string* output) {
	text_to_phonemes(text.c_str(), text.size(), output);
}

struct word_phonemizer * word_phonemizer_from_gguf(gguf_context * meta) {
	struct single_pass_tokenizer * tokenizer = single_pass_tokenizer_from_gguf(meta);
	word_phonemizer * wph = new word_phonemizer(tokenizer);
    int rule_keys_key = gguf_find_key(meta, "phonemizer.rules.keys");
    int phoneme_key = gguf_find_key(meta, "phonemizer.rules.phonemes");
    if (rule_keys_key == -1 || phoneme_key == -1) {
    	TTS_ABORT("Both 'phonemizer.rules.keys' and 'phonemizer.rules.phonemes' keys must be set in order to support phonemization.");
    }
    int key_count = gguf_get_arr_n(meta, rule_keys_key);
    assert(key_count == gguf_get_arr_n(meta, phoneme_key));
    for (int i = 0; i < key_count; i++) {
    	std::string rule_key = gguf_get_arr_str(meta, rule_keys_key, i);
    	std::string phoneme = gguf_get_arr_str(meta, phoneme_key, i);
    	wph->add_rule(split(rule_key, "."), phoneme);
    }
    return wph;
}

dictionary_response * response_from_string(std::string value, std::string key) {
	std::vector<std::string> parts = split(value, ":");
	bool has_spacing = parts.size() > 1;
	bool expects_to_be_proceeded_by_number = key[0] == '$';
	bool not_at_start = key[0] == '#';
	bool not_at_end = key.back() == '#';
    if (!has_spacing) {
    	dictionary_response * resp = new dictionary_response(SUCCESS_TOTAL, value);
    	resp->expects_to_be_proceeded_by_number = expects_to_be_proceeded_by_number;
    	resp->not_at_clause_end = not_at_end;
    	resp->not_at_clause_start = not_at_start;
    	return resp;
    } else {
    	dictionary_response * resp = new dictionary_response(SUCCESS_PARTIAL, parts[0]);
    	resp->after_match = parts[1];
    	resp->expects_to_be_proceeded_by_number = expects_to_be_proceeded_by_number;
    	resp->not_at_clause_end = not_at_end;
    	resp->not_at_clause_start = not_at_start;
    	return resp;
    }
}

struct phoneme_dictionary * phoneme_dictionary_from_gguf(gguf_context * meta) {
	struct phoneme_dictionary * dict = new phoneme_dictionary;

    int keys_key = gguf_find_key(meta, "phonemizer.dictionary.keys");
    int values_key = gguf_find_key(meta, "phonemizer.dictionary.values");
    if (keys_key == -1 || values_key == -1) {
    	TTS_ABORT("Both 'phonemizer.dictionary.keys' and 'phonemizer.dictionary.values' keys must be set in order to support phonemization.");
    }
    int key_count = gguf_get_arr_n(meta, keys_key);
    assert(key_count == gguf_get_arr_n(meta, values_key));
    for (int i = 0; i < key_count; i++) {
    	std::string key = gguf_get_arr_str(meta, keys_key, i);
    	std::string values = gguf_get_arr_str(meta, values_key, i);
    	std::vector<dictionary_response*> out;
    	for (std::string val : split(values, ",")) {
    		out.push_back(response_from_string(val, key));
    	}
    	if (key[0] == '$' || key[0] == '#') {
    		key = key.substr(1);
    	}
    	if (key.back() == '#') {
    		key = key.substr(0, key.size() - 1);
    	}
    	dict->lookup_map[key] = out;
    }
    return dict;
}

struct phonemizer * phonemizer_from_gguf(gguf_context * meta, const std::string espeak_voice_code) {
	int mode_key = gguf_find_key(meta, "phonemizer.type");
	phonemizer * ph;
    if (mode_key == -1) {
        TTS_ABORT("Key 'phonemizer.type' must be specified in gguf file for all models using a phonemizer.");
    }
    uint32_t ph_type = gguf_get_val_u32(meta, mode_key);

    if ((phonemizer_type) ph_type == ESPEAK) {
#ifdef ESPEAK_INSTALL
    	espeak_wrapper::get_instance()->initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, ESPEAK_DATA_PATH, 0);

    	update_voice(espeak_voice_code);

		ph = new phonemizer(nullptr, nullptr);
		ph->mode = ESPEAK;
#else
		TTS_ABORT("%s attempted to load an espeak phonemizer without espeak installed. \n", __func__);
#endif
		int phoneme_type_key = gguf_find_key(meta, "phonemizer.phoneme_type");
		if (phoneme_type_key != -1) {
			uint32_t phoneme_typing = gguf_get_val_u32(meta, mode_key);
			if ((phoneme_type)phoneme_typing == ESPEAK_PHONEMES) {
				ph->phoneme_mode = ESPEAK_PHONEMES;
			}
		}
		return ph;
    }
    struct word_phonemizer * phonetic_ph = word_phonemizer_from_gguf(meta);
    struct phoneme_dictionary * dict =  phoneme_dictionary_from_gguf(meta);
    ph = new phonemizer(dict, phonetic_ph);
    return ph;
}

struct phonemizer * espeak_phonemizer(bool use_espeak_phonemes, std::string espeak_voice_code) {
#ifdef ESPEAK_INSTALL
	espeak_wrapper::get_instance()->initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, ESPEAK_DATA_PATH, 0);

	update_voice(espeak_voice_code);

	phonemizer * ph = new phonemizer(nullptr, nullptr);
	ph->mode = ESPEAK;
	if (use_espeak_phonemes) {
		ph->phoneme_mode = ESPEAK_PHONEMES;
	}
	return ph;
#else
	TTS_ABORT("%s attempted to load an espeak phonemizer without espeak installed. \n", __func__);
#endif
}

struct phonemizer * phonemizer_from_file(const std::string fname, const std::string espeak_voice_code) {
	ggml_context * weight_ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!meta_ctx) {
        TTS_ABORT("%s failed for file %s\n", __func__, fname.c_str());
    }
    return phonemizer_from_gguf(meta_ctx, espeak_voice_code);
}
