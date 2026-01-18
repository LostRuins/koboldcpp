#ifndef phonemizer_h
#define phonemizer_h

#ifdef ESPEAK_INSTALL
# ifdef ESPEAK_INSTALL_LOCAL
#  include "speak_lib.h"
# else
#  include <espeak-ng/speak_lib.h>
# endif
#endif

#include <unordered_map>
#include <map>
#include <unordered_set>
#include "ttstokenizer.h"
#include <algorithm>
#include <mutex>

static const std::string ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const std::string ACCENTED_A = "àãâäáåÀÃÂÄÁÅ";
static const std::string ACCENTED_C = "çÇ";
static const std::string ACCENTED_E = "èêëéÈÊËÉ";
static const std::string ACCENTED_I = "ìîïíÌÎÏÍ";
static const std::string ACCENTED_N = "ñÑ";
static const std::string ACCENTED_O = "òõôöóøÒÕÔÖÓØ";
static const std::string ACCENTED_U = "ùûüúÙÛÜÚ";
static const std::string COMMON_ACCENTED_CHARACTERS = ACCENTED_A + ACCENTED_C + ACCENTED_E + ACCENTED_I + ACCENTED_N + ACCENTED_O + ACCENTED_U;
static const std::string WORD_CHARACTERS = ALPHABET + "." + COMMON_ACCENTED_CHARACTERS;
static const std::string NON_CLAUSE_WORD_CHARACTERS = ALPHABET + COMMON_ACCENTED_CHARACTERS + "'";
static const std::string VOWELS = "aeiouy";
static const std::unordered_set<std::string> ONE_LETTER_WORDS = {
	"a",
	"i",
};
/*
 * The two letter and three letter words listed below have been filtered down from the complete list of english two and three letter words
 * via several criteria:
 *   1. All non-EN-US words have been removed
 * 	 2. All three letter acronyms have been removed (as these lists are used to identify acronyms)
 *   3. All archaic, deprecated, or poetic words have been removed.
 * 	 4. All literary, abbreviative, and slang words have been removed if they see no more than a mean of 30 daily searches via google (over the
 *	 last 10 years).
 *
 * After the lists were filtered by the criteria described above, removed items were reviewed. Any item which had entered the common EN-US
 * vernacular but was not identified as of American origin was reintroduced into the sets below.
 */
static const std::unordered_set<std::string> TWO_LETTER_WORDS = {
	"ab", "ah", "am", "an", "as", "at", "aw", "ax", "ay", "be", "bo", "br",
	"by", "do", "eh", "er", "ew", "ex", "go", "ha", "he", "hi", "hm", "ho",
	"id", "if", "in", "is", "it", "la", "lo", "ma", "me", "mm", "my", "na",
	"no", "of", "oh", "oi", "on", "oo", "or", "ow", "ox", "oy", "pa", "qi",
	"re", "sh", "so", "to", "uh", "um", "un", "up", "us", "we", "wo", "ya",
	"ye", "yo",
};
static const std::unordered_set<std::string> THREE_LETTER_WORDS = {
	"aah", "abs", "aby", "ace", "ach", "ack", "act", "add", "ado", "ads", "aft", "age",
	"ago", "aha", "ahi", "aid", "ail", "aim", "air", "alb", "ale", "all", "alp", "alt",
	"ama", "amp", "and", "ant", "any", "ape", "app", "apt", "arc", "are", "arf", "ark",
	"arm", "art", "ash", "ask", "asp", "ass", "ate", "awe", "axe", "aye", "baa", "bad",
	"bae", "bag", "bah", "bam", "ban", "bao", "bap", "bar", "bat", "bay", "bed", "bee",
	"beg", "bet", "bez", "bib", "bid", "big", "bin", "bio", "bis", "bit", "biz", "boa",
	"bod", "bog", "boi", "boo", "bop", "bot", "bow", "box", "boy", "bra", "bro", "brr",
	"bub", "bud", "bug", "bum", "bun", "bur", "bus", "but", "buy", "bye", "cab", "caf",
	"cam", "can", "cap", "car", "cat", "caw", "chi", "cig", "cis", "cly", "cob", "cod",
	"cog", "col", "con", "coo", "cop", "cos", "cot", "cow", "cox", "coy", "cry", "cub",
	"cue", "cum", "cup", "cur", "cut", "cuz", "dab", "dad", "dag", "dal", "dam", "dap",
	"das", "daw", "day", "deb", "def", "del", "den", "dep", "dew", "dib", "did", "die",
	"dif", "dig", "dim", "din", "dip", "dis", "div", "doc", "doe", "dog", "doh", "dom",
	"don", "dos", "dot", "dox", "dry", "dub", "dud", "due", "dug", "duh", "dum", "dun",
	"duo", "dup", "dur", "dye", "ear", "eat", "ebb", "eco", "eek", "eel", "egg", "ego",
	"elf", "elk", "elm", "emo", "emu", "end", "eon", "era", "err", "est", "eve", "eww",
	"eye", "fab", "fad", "fae", "fag", "fah", "fam", "fan", "fap", "far", "fat", "fav",
	"fax", "fay", "fed", "fee", "feh", "fem", "fen", "few", "fey", "fez", "fib", "fid",
	"fig", "fin", "fir", "fit", "fix", "flu", "fly", "fob", "foe", "fog", "foo", "fop",
	"for", "fox", "fro", "fry", "fub", "fun", "fur", "gab", "gad", "gag", "gal", "gam",
	"gap", "gas", "gay", "gee", "gel", "gem", "gen", "geo", "get", "gib", "gid", "gif",
	"gig", "gin", "gip", "git", "goa", "gob", "god", "goo", "gor", "got", "gov", "grr",
	"gum", "gun", "gup", "gut", "guy", "gym", "gyp", "had", "hag", "hah", "haj", "ham",
	"hap", "has", "hat", "haw", "hay", "heh", "hem", "hen", "her", "hes", "hew", "hex",
	"hey", "hic", "hid", "him", "hip", "his", "hit", "hmm", "hod", "hoe", "hog", "hop",
	"hot", "how", "hoy", "hub", "hue", "hug", "huh", "hum", "hun", "hup", "hut", "ice",
	"ich", "ick", "icy", "ids", "ifs", "ill", "imp", "ink", "inn", "int", "ion", "ire",
	"irk", "ism", "its", "ivy", "jab", "jam", "jap", "jar", "jaw", "jay", "jet", "jib",
	"jig", "jin", "job", "joe", "jog", "jot", "joy", "jug", "jut", "kat", "kaw", "kay",
	"ked", "keg", "key", "kid", "kin", "kit", "kob", "koi", "lab", "lac", "lad", "lag",
	"lam", "lap", "law", "lax", "lay", "led", "leg", "lei", "lek", "let", "lev", "lex",
	"lib", "lid", "lie", "lip", "lit", "lob", "log", "loo", "lop", "lot", "low", "lug",
	"luv", "lye", "mac", "mad", "mag", "mam", "man", "map", "mar", "mat", "maw", "max",
	"may", "med", "meg", "meh", "mel", "men", "met", "mew", "mib", "mid", "mig", "mil",
	"mix", "mmm", "mob", "mod", "mog", "mol", "mom", "mon", "moo", "mop", "mow", "mud",
	"mug", "mum", "mut", "nab", "nag", "nah", "nan", "nap", "nat", "naw", "nay", "nef",
	"neg", "net", "new", "nib", "nil", "nip", "nit", "nob", "nod", "nog", "noh", "nom",
	"non", "noo", "nor", "not", "now", "noy", "nth", "nub", "nun", "nut", "nyx", "oaf",
	"oak", "oar", "oat", "oba", "obs", "oca", "odd", "ode", "off", "oft", "ohm", "oil",
	"oke", "old", "one", "oof", "ooh", "oom", "oop", "ops", "opt", "orb", "orc", "ore",
	"org", "ort", "oud", "our", "out", "ova", "owe", "owl", "own", "oxy", "pad", "pah",
	"pal", "pan", "par", "pas", "pat", "paw", "pax", "pay", "pea", "pec", "pee", "peg",
	"pen", "pep", "per", "pes", "pet", "pew", "phi", "pho", "pht", "pic", "pie", "pig",
	"pin", "pip", "pit", "pix", "ply", "pod", "poi", "pol", "poo", "pop", "pos", "pot",
	"pow", "pox", "pre", "pro", "pry", "psi", "pst", "pub", "pug", "puh", "pul", "pun",
	"pup", "pur", "pus", "put", "pwn", "pya", "pyx", "qat", "rad", "rag", "rai", "raj",
	"ram", "ran", "rap", "rat", "raw", "ray", "reb", "rec", "red", "ref", "reg", "rem",
	"res", "ret", "rex", "rez", "rho", "ria", "rib", "rid", "rig", "rim", "rin", "rip",
	"rob", "roc", "rod", "roe", "rom", "rot", "row", "rub", "rue", "rug", "rum", "run",
	"rut", "rya", "rye", "sac", "sad", "sag", "sal", "sap", "sat", "saw", "sax", "say",
	"sea", "sec", "see", "seg", "sen", "set", "sew", "sex", "she", "shh", "shy", "sib",
	"sic", "sig", "sim", "sin", "sip", "sir", "sis", "sit", "six", "ska", "ski", "sky",
	"sly", "sob", "sod", "sol", "som", "son", "sop", "sot", "sou", "sow", "sox", "soy",
	"spa", "spy", "sty", "sub", "sue", "sum", "sun", "sup", "sus", "tab", "tad", "tag",
	"tai", "taj", "tan", "tao", "tap", "tar", "tat", "tau", "tav", "taw", "tax", "tea",
	"tec", "tee", "teg", "tel", "ten", "tet", "tex", "the", "tho", "thy", "tic", "tie",
	"til", "tin", "tip", "tis", "tit", "tod", "toe", "ton", "too", "top", "tor", "tot",
	"tow", "toy", "try", "tsk", "tub", "tug", "tui", "tum", "tun", "tup", "tut", "tux",
	"two", "ugh", "umm", "ump", "uni", "ups", "urd", "urn", "use", "uta", "ute", "utu",
	"uwu", "vac", "van", "var", "vas", "vat", "vav", "vax", "vee", "veg", "vet", "vex",
	"via", "vid", "vie", "vig", "vim", "vol", "vow", "vox", "vug", "wad", "wag", "wan",
	"wap", "war", "was", "wat", "wax", "way", "web", "wed", "wee", "wen", "wet", "wey",
	"who", "why", "wig", "win", "wit", "wiz", "woe", "wok", "won", "woo", "wop", "wow",
	"wry", "wud", "wus", "yag", "yah", "yak", "yam", "yap", "yar", "yaw", "yay", "yea",
	"yeh", "yen", "yep", "yes", "yet", "yew", "yin", "yip", "yok", "you", "yow", "yum",
	"yup", "zag", "zap", "zax", "zed", "zee", "zen", "zig", "zip", "zit", "zoo", "zzz"
};

static const std::map<const char, std::string> LETTER_PHONEMES = {
	{'a', "ˈeɪ"},
	{'b', "bˈiː"},
	{'c', "sˈiː"},
	{'d', "dˈiː"},
	{'e', "ˈiː"},
	{'f', "ˈɛf"},
	{'g', "dʒˈi"},
	{'h', "ˈeɪtʃ"},
	{'i', "ˈaɪ"},
	{'j', "dʒˈeɪ"},
	{'k', "kˈeɪ"},
	{'l', "ˈɛl"},
	{'m', "ˈɛm"},
	{'n', "ˈɛn"},
	{'o', "ˈoʊ"},
	{'p', "pˈiː"},
	{'q', "kjˈuː"},
	{'r', "ˈɑːɹ"},
	{'s', "ˈɛs"},
	{'t', "tˈiː"},
	{'u', "jˈuː"},
	{'v', "vˈiː"},
	{'w', "dˈʌbəljˌuː"},
	{'x', "ˈɛks"},
	{'y', "wˈaɪ"},
	{'z', "zˈiː"}
};

static const std::string SPACE_CHARACTERS = " \t\f\n";
static const std::string NOOP_BREAKS = "{}[]():;,\"";
static const std::string CLAUSE_BREAKS = ".!?";

static const std::string TRILLION_PHONEME = "tɹˈɪliən";
static const long long int TRILLION = 1000000000000;
static const std::string BILLION_PHONEME = "bˈɪliən";
static const int BILLION = 1000000000;
static const std::string MILLION_PHONEME = "mˈɪliən";
static const int MILLION = 1000000;
static const std::string POINT_PHONEME = "pˈɔɪnt";
static const std::string THOUSAND_PHONEME = "θˈaʊzənd";
static const std::string HUNDRED_PHONEME = "hˈʌndɹɪd";
static const std::string NUMBER_CHARACTERS = "0123456789";
static const std::string COMPATIBLE_NUMERICS = NUMBER_CHARACTERS + "., ";
static const long long int LARGEST_PRONOUNCABLE_NUMBER = 999999999999999;

static const std::vector<std::string> NUMBER_PHONEMES = {
	"zˈiəɹoʊ",
	"wˈʌn",
	"tˈuː",
	"θɹˈiː",
	"fˈɔːɹ",
	"fˈaɪv",
	"sˈɪks",
	"sˈɛvən",
	"ˈeɪt",
	"nˈaɪn",
	"tˈɛn",
	"ɪlˈɛvən",
	"twˈɛlv",
	"θˈɜːtiːn",
	"fˈɔːɹtiːn",
	"fˈɪftiːn",
	"sˈɪkstiːn",
	"sˈɛvəntˌiːn",
	"ˈeɪtiːn",
	"nˈaɪntiːn"
};

static const std::vector<std::string> SUB_HUNDRED_NUMBERS = {
	"twˈɛnti",
	"θˈɜːɾi",
	"fˈɔːɹɾi",
	"fˈɪfti",
	"sˈɪksti",
	"sˈɛvənti",
	"ˈeɪɾi",
	"nˈaɪnti"
};

static const std::map<std::string, std::string> REPLACEABLE = {
	{"*", "ˈæstɚɹˌɪsk"},
	{"+", "plˈʌs"},
	{"&", "ˈænd"},
	{"%", "pɚsˈɛnt"},
	{"@", "ˈæt"},
	{"#", "hˈæʃ"},
	{"$", "dˈɑːlɚ"},
	{"~", "tˈɪldə"},
	{"¢", "sˈɛnts"},
	{"£", "pˈaʊnd"},
	{"¥", "jˈɛn"},
	{"₨", "ɹˈuːpiː"},
	{"€", "jˈʊɹɹoʊz"},
	{"₹", "ɹˈuːpiː"},
	{"♯", "ʃˈɑːɹp"},
	{"♭", "flˈæt"},
	{"≈", "ɐpɹˈɑːksɪmətli"},
	{"≠", "nˈɑːt ˈiːkwəl tʊ"},
	{"≤", "lˈɛs ɔːɹ ˈiːkwəl tʊ"},
	{"≥", "ɡɹˈeɪɾɚɹ ɔːɹ ˈiːkwəl tʊ"},
	{">", "ɡɹˈeɪɾɚ ðɐn"},
	{"<", "lˈɛs ðɐn"},
	{"=", "ˈiːkwəlz"},
	{"±", "plˈʌs ɔːɹ mˈaɪnəs"},
	{"×", "tˈaɪmz"},
	{"÷", "dᵻvˈaɪdᵻd bˈaɪ"},
	{"℞", "pɹɪskɹˈɪpʃən"},
	{"№", "nˈuːməˌoʊ"},
	{"°", "dᵻɡɹˈiːz"},
	{"∴", "ðˈɛɹfɔːɹ"},
	{"∵", "bɪkˈʌz"},
	{"√", "skwˈɛɹ ɹˈuːt"},
	{"∛", "kjˈuːb ɹˈuːt"},
	{"∑", "sˈʌm sˈaɪn"},
	{"∂", "dˈɛltə"},
	{"←", "lˈɛft ˈæɹoʊ"},
	{"↑", "ˈʌp ˈæɹoʊ"},
	{"→", "ɹˈaɪt ˈæɹoʊ"},
	{"↓", "dˈaʊn ˈæɹoʊ"},
	{"−", "mˈaɪnəs"},
	{"¶", "pˈæɹəɡɹˌæf"},
	{"§", "sˈɛkʃən"},
};

static const std::string ROMAN_NUMERAL_CHARACTERS = "MDCLXVImdclxvi";
static const std::map<std::string, int> ROMAN_NUMERALS = {
	{"m", 1000},
	{"mm", 2000},
	{"mmm", 3000},
	{"c", 100},
	{"cc", 200},
	{"ccc", 300},
	{"cd", 400},
	{"cm", 900},
	{"dc", 600},
	{"dcc", 700},
	{"dccc", 800},
	{"x", 10},
	{"xx", 20},
	{"xxx", 30},
	{"xl", 40},
	{"l", 50},
	{"lx", 60},
	{"lxx", 70},
	{"lxxx", 80},
	{"xc", 90},
	{"i", 1},
	{"ii", 2},
	{"iii", 3},
	{"iv", 4},
	{"v", 5},
	{"vi", 6},
	{"vii", 7},
	{"viii", 8},
	{"ix", 9},
};

static const std::map<std::string, std::string> CONTRACTION_PHONEMES = {
	{"re", "r"},
	{"ve", "əv"},
	{"ll", "l"},
	{"d", "d"},
	{"t", "t"},
};

// characters that Espeak-ng treats as stopping tokens.
static std::string STOPPING_TOKENS = ".,:;!?";

#ifdef ESPEAK_INSTALL
/**
 * espeak-ng uses globals to persist and manage its state so it is not compatible with
 * threaded parallelism (https://github.com/espeak-ng/espeak-ng/issues/1527).
 * This singleton acts as a mutex wrapped provider for all espeak phonemization methods such
 * that multiple instances of the kokoro_runner can be initialized and called in parallel.
 */
class espeak_wrapper {
private:
    static espeak_wrapper * instance;
    static std::mutex mutex;

protected:
    espeak_wrapper() {};
    ~espeak_wrapper() {};
    bool espeak_initialized = false;

public:
	// singletons aren't copyable
    espeak_wrapper(espeak_wrapper &other) = delete;

    // singletons aren't assignable
    void operator=(const espeak_wrapper &) = delete;

    static espeak_wrapper * get_instance();
    const espeak_VOICE ** list_voices();
    espeak_ERROR set_voice(const char * voice_code);
    const char * text_to_phonemes(const void ** textptr, int textmode, int phonememode);
    void initialize(espeak_AUDIO_OUTPUT output, int buflength, const char * path, int options);
};
#endif

enum lookup_code {
	SUCCESS_TOTAL = 100,
	SUCCESS_PARTIAL = 101,
	FAILURE_UNFOUND = 200,
	FAILURE_PHONETIC = 201,
};

enum phoneme_type {
	IPA = 1,
	ESPEAK_PHONEMES = 2,
};

enum phonemizer_type {
	TTS_PHONEMIZER = 0,
	ESPEAK = 1,
};

std::string parse_voice_code(std::string voice_code);
void update_voice(std::string voice_code);
const std::unordered_set<std::string> inline_combine_sets(const std::vector<std::unordered_set<std::string>> sets);
int upper_count(std::string word);
bool is_all_upper(std::string word);
bool is_roman_numeral(char letter);
bool can_be_roman_numeral(std::string word);
bool is_alphabetic(char letter);
bool is_numeric(char letter);


std::string replace_accents(std::string word);
std::string build_subthousand_phoneme(int value);
std::string build_number_phoneme(long long int remainder);

// The conditions struct is used to track and describe stateful criteria while converting text to phonemes.
struct conditions {
	bool hyphenated = false;
	bool was_all_capitalized = false;
	bool was_word = false;
	bool was_punctuated_acronym = false;
	bool was_number = false;
	bool beginning_of_clause = true;

	void reset_for_clause_end();
	void reset_for_space();
	void update_for_word(std::string word,bool allow_for_upper_check = true);
};

/*
 * The corpus struct is simply a small wrapper class that is used to perform simple look forward and backwards in the text
 * which is being phonemized. This can be used to discern how to convert chunks of text in a consistent and protective fashion
 * in order to accurately phonemize complicated text.
 */
struct corpus {
	corpus(const char * text, size_t size): size(size), text(text) {};
	size_t location = 0;
	size_t size;
	const char * text;

	/*
	 * These all return strings because we are parsing in utf-8. As such the count variables passed to all the functions do not represent
	 * the byte offset to pull to but rather the number of full utf-8 characters to pull (this can include 2, 3, and 4 byte characters).
	 */
	std::string next(int count = 1);
	std::string last(int count = 1);
	std::string pop(int count = 1);
	std::string after(int after = 1, int count = 1);

	// this is used for popping byte count rather than unique character count.
	std::string size_pop(size_t pop_size);

	std::string next_in(std::string val, bool* has_accent = nullptr);
	std::string pop_in(std::string val);

	std::string after_until(int after, std::string val);
};

/*
 * The TTS phonemizer works by splitting each word into distinct graphemes, and for each grapheme the phonemizer will look at the grapheme that came
 * before, after, and for any word specific exceptions in order to compile a
 */
struct phonemizer_rule {
	~phonemizer_rule() {
		for (auto it : rules) {
			delete it.second;
		}
	}

	std::unordered_map<std::string, phonemizer_rule*> rules;
	std::string value = "";
	std::string lookup_rule(std::vector<std::string> & keys, int index);
};

typedef std::unordered_map<std::string, phonemizer_rule*> rules_lookup;

struct word_phonemizer {
	word_phonemizer(struct single_pass_tokenizer * tokenizer): tokenizer(tokenizer) {};
	~word_phonemizer() {
		for (auto it : rules) {
			delete it.second;
		}
		delete tokenizer;
	}

	struct single_pass_tokenizer * tokenizer;
	rules_lookup rules;

	std::string phonemize(std::string word);
	void add_rule(std::vector<std::string> keys, std::string phoneme);

private:
	std::string lookup_rule(std::string word, std::string current_token, std::string last_token, std::string next_token);
};

struct word_phonemizer * word_phonemizer_from_gguf(gguf_context * meta);

/*
 * The general translation approach that espeak uses is to lookup words in the dictionary and return a list of possible matches per lookup.
 * Each match contains flags which describe the match's conditions and limitations and optionally a pronunciation. When a pronunciation is not returned,
 * it usually means that the word needs to be pronounced phonetically, the word belongs to another language, or that the original content is a
 * token representation of a different word (e.g. with numbers).
 *
 * Since it does not make sense to have the core lexer reperform this lookup operation with represented words or via distinct languages, those behaviors
 * are managed by the lookup operation itself and thus the lookup operation will only fail when phonetic or acronym content should be produced.
 */
struct dictionary_response {
	dictionary_response(lookup_code code, std::string value = ""): code(code), value(value) {}
	std::string value;
	lookup_code code;
	bool expects_to_be_proceeded_by_number = false;
	bool not_at_clause_end = false;
	bool not_at_clause_start = false;

	std::string after_match = "";

	bool is_successful();
	bool is_match(corpus* text, conditions* flags);
};

dictionary_response * response_from_string(std::string value, std::string key);

struct phoneme_dictionary {
	std::unordered_map<std::string, std::vector<dictionary_response*>> lookup_map;
	dictionary_response* lookup(corpus* text,std::string value, conditions* flags);
	dictionary_response* not_found_response = new dictionary_response(FAILURE_UNFOUND);
	dictionary_response* phonetic_fallback_response = new dictionary_response(FAILURE_PHONETIC);
};

struct phoneme_dictionary * phoneme_dictionary_from_gguf(gguf_context * meta);

/*
 * In general, I would like to avoid requiring the installation of otherwise broad and technically complicated libraries,
 * like espeak, especially when they are only being used for a small portion of their overall functionality. While avoiding these
 * requirements will keep the default installation cost of TTS.cpp down, it is also unlikely that TTS.cpp will support
 * the level of variability in phonemization that espeak currently does. In this regard, I have chosen to optionally support usage of
 * espeak. As such, the phonemizer struct described below will support simple text to IPA phoneme functionality out of the box,
 * while also optionally acting as an interface for espeak phonemization.
 *
 * Phonemization seems to use a pattern close to the common lexer, such that at each index or chunk of text forward and backward context
 * views are used to support single pass translation. As such, the TTS.cpp phonemization pattern I've decided to implement behaves
 * effecively like a simple router lexer. It will only support utf-8 encoded text and english IPA conversion.
 */
struct phonemizer {
	phonemizer(struct phoneme_dictionary * dict, struct word_phonemizer * phonetic_phonemizer, bool preserve_punctuation = true): dict(dict), phonetic_phonemizer(phonetic_phonemizer), preserve_punctuation(preserve_punctuation) {};
	~phonemizer() {
		delete dict;
		delete phonetic_phonemizer;
	}
	const std::unordered_set<std::string> small_english_words = inline_combine_sets({THREE_LETTER_WORDS, TWO_LETTER_WORDS, ONE_LETTER_WORDS});
	std::string separator = " ";
	phoneme_type phoneme_mode = IPA;
	phonemizer_type mode = TTS_PHONEMIZER;
	bool preserve_punctuation = true;

	struct phoneme_dictionary * dict;

	struct word_phonemizer * phonetic_phonemizer;

	void text_to_phonemes(std::string text, std::string* output);
	void text_to_phonemes(const char * text, size_t size, std::string* output);
	std::string text_to_phonemes(std::string text);
	std::string text_to_phonemes(const char * text, size_t size);

#ifdef ESPEAK_INSTALL
	std::string espeak_text_to_phonemes(const char * text);
#endif

	bool process_word(corpus* text, std::string* output, std::string word, conditions * flags, bool has_accent = false);
	void append_numeric_series(std::string series, std::string* output, conditions * flags);
	bool is_acronym_like(corpus* text, std::string word, conditions* flags);

	bool route(corpus* text, std::string* output, conditions* flags);
	bool handle_space(corpus* text, std::string* output, conditions* flags);
	bool handle_contraction(corpus* text, std::string* output, conditions* flags);
	bool handle_possession_plural(corpus* text, std::string* output, conditions* flags);
	bool handle_replacement(corpus* text, std::string next, std::string* output, conditions * flags);
	bool handle_phonetic(corpus* text, std::string word, std::string* output, conditions* flags, size_t unaccented_size_difference);
	bool handle_acronym(corpus* text, std::string word, std::string* output, conditions * flags);
	bool handle_roman_numeral(corpus* text, std::string* output, conditions * flags);
	bool handle_word(corpus* text, std::string* output, conditions* flags);
	bool handle_numeric_series(corpus* text, std::string* output, conditions* flags);
	bool handle_numeric(corpus* text, std::string* output, conditions* flags);
	bool handle_punctuation(corpus* text, std::string next, std::string* output, conditions* flags);
	bool handle_unknown(corpus* text);
};

struct phonemizer * phonemizer_from_gguf(gguf_context * meta, const std::string espeak_voice_code = "gmw/en-US");
struct phonemizer * phonemizer_from_file(const std::string fname, const std::string espeak_voice_code = "gmw/en-US");
struct phonemizer * espeak_phonemizer(bool use_espeak_phonemes = false, std::string espeak_voice_code = "gmw/en-US");

#endif
