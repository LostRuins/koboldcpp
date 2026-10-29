#include "engine/models/cosyvoice3/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::cosyvoice3 {
namespace {

constexpr int32_t kEndOfPromptId = 151646;
constexpr std::string_view kEndOfPromptText = "<|endofprompt|>";
constexpr std::string_view kDefaultPromptPrefix = "You are a helpful assistant.";

std::vector<engine::tokenizers::LlamaBpeAddedToken> cosyvoice3_special_tokens() {
    std::vector<engine::tokenizers::LlamaBpeAddedToken> tokens;
    tokens.reserve(280);
    tokens.emplace_back("<|im_start|>", 151644);
    tokens.emplace_back("<|im_end|>", 151645);
    tokens.emplace_back("<|endofprompt|>", 151646);
    tokens.emplace_back("[breath]", 151647);
    tokens.emplace_back("<strong>", 151648);
    tokens.emplace_back("</strong>", 151649);
    tokens.emplace_back("[noise]", 151650);
    tokens.emplace_back("[laughter]", 151651);
    tokens.emplace_back("[cough]", 151652);
    tokens.emplace_back("[clucking]", 151653);
    tokens.emplace_back("[accent]", 151654);
    tokens.emplace_back("[quick_breath]", 151655);
    tokens.emplace_back("<laughter>", 151656);
    tokens.emplace_back("</laughter>", 151657);
    tokens.emplace_back("[hissing]", 151658);
    tokens.emplace_back("[sigh]", 151659);
    tokens.emplace_back("[vocalized-noise]", 151660);
    tokens.emplace_back("[lipsmack]", 151661);
    tokens.emplace_back("[mn]", 151662);
    tokens.emplace_back("<|endofsystem|>", 151663);
    tokens.emplace_back("[AA]", 151664);
    tokens.emplace_back("[AA0]", 151665);
    tokens.emplace_back("[AA1]", 151666);
    tokens.emplace_back("[AA2]", 151667);
    tokens.emplace_back("[AE]", 151668);
    tokens.emplace_back("[AE0]", 151669);
    tokens.emplace_back("[AE1]", 151670);
    tokens.emplace_back("[AE2]", 151671);
    tokens.emplace_back("[AH]", 151672);
    tokens.emplace_back("[AH0]", 151673);
    tokens.emplace_back("[AH1]", 151674);
    tokens.emplace_back("[AH2]", 151675);
    tokens.emplace_back("[AO]", 151676);
    tokens.emplace_back("[AO0]", 151677);
    tokens.emplace_back("[AO1]", 151678);
    tokens.emplace_back("[AO2]", 151679);
    tokens.emplace_back("[AW]", 151680);
    tokens.emplace_back("[AW0]", 151681);
    tokens.emplace_back("[AW1]", 151682);
    tokens.emplace_back("[AW2]", 151683);
    tokens.emplace_back("[AY]", 151684);
    tokens.emplace_back("[AY0]", 151685);
    tokens.emplace_back("[AY1]", 151686);
    tokens.emplace_back("[AY2]", 151687);
    tokens.emplace_back("[B]", 151688);
    tokens.emplace_back("[CH]", 151689);
    tokens.emplace_back("[D]", 151690);
    tokens.emplace_back("[DH]", 151691);
    tokens.emplace_back("[EH]", 151692);
    tokens.emplace_back("[EH0]", 151693);
    tokens.emplace_back("[EH1]", 151694);
    tokens.emplace_back("[EH2]", 151695);
    tokens.emplace_back("[ER]", 151696);
    tokens.emplace_back("[ER0]", 151697);
    tokens.emplace_back("[ER1]", 151698);
    tokens.emplace_back("[ER2]", 151699);
    tokens.emplace_back("[EY]", 151700);
    tokens.emplace_back("[EY0]", 151701);
    tokens.emplace_back("[EY1]", 151702);
    tokens.emplace_back("[EY2]", 151703);
    tokens.emplace_back("[F]", 151704);
    tokens.emplace_back("[G]", 151705);
    tokens.emplace_back("[HH]", 151706);
    tokens.emplace_back("[IH]", 151707);
    tokens.emplace_back("[IH0]", 151708);
    tokens.emplace_back("[IH1]", 151709);
    tokens.emplace_back("[IH2]", 151710);
    tokens.emplace_back("[IY]", 151711);
    tokens.emplace_back("[IY0]", 151712);
    tokens.emplace_back("[IY1]", 151713);
    tokens.emplace_back("[IY2]", 151714);
    tokens.emplace_back("[JH]", 151715);
    tokens.emplace_back("[K]", 151716);
    tokens.emplace_back("[L]", 151717);
    tokens.emplace_back("[M]", 151718);
    tokens.emplace_back("[N]", 151719);
    tokens.emplace_back("[NG]", 151720);
    tokens.emplace_back("[OW]", 151721);
    tokens.emplace_back("[OW0]", 151722);
    tokens.emplace_back("[OW1]", 151723);
    tokens.emplace_back("[OW2]", 151724);
    tokens.emplace_back("[OY]", 151725);
    tokens.emplace_back("[OY0]", 151726);
    tokens.emplace_back("[OY1]", 151727);
    tokens.emplace_back("[OY2]", 151728);
    tokens.emplace_back("[P]", 151729);
    tokens.emplace_back("[R]", 151730);
    tokens.emplace_back("[S]", 151731);
    tokens.emplace_back("[SH]", 151732);
    tokens.emplace_back("[T]", 151733);
    tokens.emplace_back("[TH]", 151734);
    tokens.emplace_back("[UH]", 151735);
    tokens.emplace_back("[UH0]", 151736);
    tokens.emplace_back("[UH1]", 151737);
    tokens.emplace_back("[UH2]", 151738);
    tokens.emplace_back("[UW]", 151739);
    tokens.emplace_back("[UW0]", 151740);
    tokens.emplace_back("[UW1]", 151741);
    tokens.emplace_back("[UW2]", 151742);
    tokens.emplace_back("[V]", 151743);
    tokens.emplace_back("[W]", 151744);
    tokens.emplace_back("[Y]", 151745);
    tokens.emplace_back("[Z]", 151746);
    tokens.emplace_back("[ZH]", 151747);
    tokens.emplace_back("[a]", 151748);
    tokens.emplace_back("[ai]", 151749);
    tokens.emplace_back("[an]", 151750);
    tokens.emplace_back("[ang]", 151751);
    tokens.emplace_back("[ao]", 151752);
    tokens.emplace_back("[b]", 151753);
    tokens.emplace_back("[c]", 151754);
    tokens.emplace_back("[ch]", 151755);
    tokens.emplace_back("[d]", 151756);
    tokens.emplace_back("[e]", 151757);
    tokens.emplace_back("[ei]", 151758);
    tokens.emplace_back("[en]", 151759);
    tokens.emplace_back("[eng]", 151760);
    tokens.emplace_back("[f]", 151761);
    tokens.emplace_back("[g]", 151762);
    tokens.emplace_back("[h]", 151763);
    tokens.emplace_back("[i]", 151764);
    tokens.emplace_back("[ian]", 151765);
    tokens.emplace_back("[in]", 151766);
    tokens.emplace_back("[ing]", 151767);
    tokens.emplace_back("[iu]", 151768);
    tokens.emplace_back("[ià]", 151769);
    tokens.emplace_back("[iàn]", 151770);
    tokens.emplace_back("[iàng]", 151771);
    tokens.emplace_back("[iào]", 151772);
    tokens.emplace_back("[iá]", 151773);
    tokens.emplace_back("[ián]", 151774);
    tokens.emplace_back("[iáng]", 151775);
    tokens.emplace_back("[iáo]", 151776);
    tokens.emplace_back("[iè]", 151777);
    tokens.emplace_back("[ié]", 151778);
    tokens.emplace_back("[iòng]", 151779);
    tokens.emplace_back("[ióng]", 151780);
    tokens.emplace_back("[iù]", 151781);
    tokens.emplace_back("[iú]", 151782);
    tokens.emplace_back("[iā]", 151783);
    tokens.emplace_back("[iān]", 151784);
    tokens.emplace_back("[iāng]", 151785);
    tokens.emplace_back("[iāo]", 151786);
    tokens.emplace_back("[iē]", 151787);
    tokens.emplace_back("[iě]", 151788);
    tokens.emplace_back("[iōng]", 151789);
    tokens.emplace_back("[iū]", 151790);
    tokens.emplace_back("[iǎ]", 151791);
    tokens.emplace_back("[iǎn]", 151792);
    tokens.emplace_back("[iǎng]", 151793);
    tokens.emplace_back("[iǎo]", 151794);
    tokens.emplace_back("[iǒng]", 151795);
    tokens.emplace_back("[iǔ]", 151796);
    tokens.emplace_back("[j]", 151797);
    tokens.emplace_back("[k]", 151798);
    tokens.emplace_back("[l]", 151799);
    tokens.emplace_back("[m]", 151800);
    tokens.emplace_back("[n]", 151801);
    tokens.emplace_back("[o]", 151802);
    tokens.emplace_back("[ong]", 151803);
    tokens.emplace_back("[ou]", 151804);
    tokens.emplace_back("[p]", 151805);
    tokens.emplace_back("[q]", 151806);
    tokens.emplace_back("[r]", 151807);
    tokens.emplace_back("[s]", 151808);
    tokens.emplace_back("[sh]", 151809);
    tokens.emplace_back("[t]", 151810);
    tokens.emplace_back("[u]", 151811);
    tokens.emplace_back("[uang]", 151812);
    tokens.emplace_back("[ue]", 151813);
    tokens.emplace_back("[un]", 151814);
    tokens.emplace_back("[uo]", 151815);
    tokens.emplace_back("[uà]", 151816);
    tokens.emplace_back("[uài]", 151817);
    tokens.emplace_back("[uàn]", 151818);
    tokens.emplace_back("[uàng]", 151819);
    tokens.emplace_back("[uá]", 151820);
    tokens.emplace_back("[uái]", 151821);
    tokens.emplace_back("[uán]", 151822);
    tokens.emplace_back("[uáng]", 151823);
    tokens.emplace_back("[uè]", 151824);
    tokens.emplace_back("[ué]", 151825);
    tokens.emplace_back("[uì]", 151826);
    tokens.emplace_back("[uí]", 151827);
    tokens.emplace_back("[uò]", 151828);
    tokens.emplace_back("[uó]", 151829);
    tokens.emplace_back("[uā]", 151830);
    tokens.emplace_back("[uāi]", 151831);
    tokens.emplace_back("[uān]", 151832);
    tokens.emplace_back("[uāng]", 151833);
    tokens.emplace_back("[uē]", 151834);
    tokens.emplace_back("[uě]", 151835);
    tokens.emplace_back("[uī]", 151836);
    tokens.emplace_back("[uō]", 151837);
    tokens.emplace_back("[uǎ]", 151838);
    tokens.emplace_back("[uǎi]", 151839);
    tokens.emplace_back("[uǎn]", 151840);
    tokens.emplace_back("[uǎng]", 151841);
    tokens.emplace_back("[uǐ]", 151842);
    tokens.emplace_back("[uǒ]", 151843);
    tokens.emplace_back("[vè]", 151844);
    tokens.emplace_back("[w]", 151845);
    tokens.emplace_back("[x]", 151846);
    tokens.emplace_back("[y]", 151847);
    tokens.emplace_back("[z]", 151848);
    tokens.emplace_back("[zh]", 151849);
    tokens.emplace_back("[à]", 151850);
    tokens.emplace_back("[ài]", 151851);
    tokens.emplace_back("[àn]", 151852);
    tokens.emplace_back("[àng]", 151853);
    tokens.emplace_back("[ào]", 151854);
    tokens.emplace_back("[á]", 151855);
    tokens.emplace_back("[ái]", 151856);
    tokens.emplace_back("[án]", 151857);
    tokens.emplace_back("[áng]", 151858);
    tokens.emplace_back("[áo]", 151859);
    tokens.emplace_back("[è]", 151860);
    tokens.emplace_back("[èi]", 151861);
    tokens.emplace_back("[èn]", 151862);
    tokens.emplace_back("[èng]", 151863);
    tokens.emplace_back("[èr]", 151864);
    tokens.emplace_back("[é]", 151865);
    tokens.emplace_back("[éi]", 151866);
    tokens.emplace_back("[én]", 151867);
    tokens.emplace_back("[éng]", 151868);
    tokens.emplace_back("[ér]", 151869);
    tokens.emplace_back("[ì]", 151870);
    tokens.emplace_back("[ìn]", 151871);
    tokens.emplace_back("[ìng]", 151872);
    tokens.emplace_back("[í]", 151873);
    tokens.emplace_back("[ín]", 151874);
    tokens.emplace_back("[íng]", 151875);
    tokens.emplace_back("[ò]", 151876);
    tokens.emplace_back("[òng]", 151877);
    tokens.emplace_back("[òu]", 151878);
    tokens.emplace_back("[ó]", 151879);
    tokens.emplace_back("[óng]", 151880);
    tokens.emplace_back("[óu]", 151881);
    tokens.emplace_back("[ù]", 151882);
    tokens.emplace_back("[ùn]", 151883);
    tokens.emplace_back("[ú]", 151884);
    tokens.emplace_back("[ún]", 151885);
    tokens.emplace_back("[ā]", 151886);
    tokens.emplace_back("[āi]", 151887);
    tokens.emplace_back("[ān]", 151888);
    tokens.emplace_back("[āng]", 151889);
    tokens.emplace_back("[āo]", 151890);
    tokens.emplace_back("[ē]", 151891);
    tokens.emplace_back("[ēi]", 151892);
    tokens.emplace_back("[ēn]", 151893);
    tokens.emplace_back("[ēng]", 151894);
    tokens.emplace_back("[ě]", 151895);
    tokens.emplace_back("[ěi]", 151896);
    tokens.emplace_back("[ěn]", 151897);
    tokens.emplace_back("[ěng]", 151898);
    tokens.emplace_back("[ěr]", 151899);
    tokens.emplace_back("[ī]", 151900);
    tokens.emplace_back("[īn]", 151901);
    tokens.emplace_back("[īng]", 151902);
    tokens.emplace_back("[ō]", 151903);
    tokens.emplace_back("[ōng]", 151904);
    tokens.emplace_back("[ōu]", 151905);
    tokens.emplace_back("[ū]", 151906);
    tokens.emplace_back("[ūn]", 151907);
    tokens.emplace_back("[ǎ]", 151908);
    tokens.emplace_back("[ǎi]", 151909);
    tokens.emplace_back("[ǎn]", 151910);
    tokens.emplace_back("[ǎng]", 151911);
    tokens.emplace_back("[ǎo]", 151912);
    tokens.emplace_back("[ǐ]", 151913);
    tokens.emplace_back("[ǐn]", 151914);
    tokens.emplace_back("[ǐng]", 151915);
    tokens.emplace_back("[ǒ]", 151916);
    tokens.emplace_back("[ǒng]", 151917);
    tokens.emplace_back("[ǒu]", 151918);
    tokens.emplace_back("[ǔ]", 151919);
    tokens.emplace_back("[ǔn]", 151920);
    tokens.emplace_back("[ǘ]", 151921);
    tokens.emplace_back("[ǚ]", 151922);
    tokens.emplace_back("[ǜ]", 151923);
    return tokens;
}

void require_end_of_prompt(const std::vector<int32_t> & tokens, const char * context) {
    for (const int32_t token : tokens) {
        if (token == kEndOfPromptId) {
            return;
        }
    }
    throw std::runtime_error(std::string("CosyVoice3 ") + context + " must contain <|endofprompt|>");
}

std::string prompt_with_boundary(std::string_view text) {
    std::string out(text);
    if (out.find(kEndOfPromptText) == std::string::npos) {
        out = std::string(kDefaultPromptPrefix) + std::string(kEndOfPromptText) + out;
    }
    return out;
}

std::string instruction_with_boundary(std::string_view instruction) {
    std::string out(instruction);
    if (out.find(kEndOfPromptText) == std::string::npos) {
        out = std::string(kDefaultPromptPrefix) + " " + out + std::string(kEndOfPromptText);
    }
    return out;
}

}  // namespace

class CosyVoice3TextTokenizer::Impl {
public:
    explicit Impl(const CosyVoice3Assets & assets) {
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.vocab_path = assets.resources.require_file("vocab_json");
        spec.merges_path = assets.resources.require_file("merges_txt");
        spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        spec.additional_special_tokens = cosyvoice3_special_tokens();
        tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    }

    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

CosyVoice3TextTokenizer::CosyVoice3TextTokenizer(std::shared_ptr<const CosyVoice3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("CosyVoice3 tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(*assets);
}

CosyVoice3TextTokenizer::~CosyVoice3TextTokenizer() = default;

CosyVoice3TextTokens CosyVoice3TextTokenizer::encode_zero_shot(
    std::string_view text,
    std::string_view prompt_text) const {
    CosyVoice3TextTokens out;
    out.prompt = impl_->tokenizer->encode(prompt_with_boundary(prompt_text), true);
    out.target = impl_->tokenizer->encode(std::string(text), true);
    require_end_of_prompt(out.prompt, "zero-shot prompt");
    return out;
}

CosyVoice3TextTokens CosyVoice3TextTokenizer::encode_cross_lingual(std::string_view text) const {
    CosyVoice3TextTokens out;
    out.target = impl_->tokenizer->encode(prompt_with_boundary(text), true);
    require_end_of_prompt(out.target, "cross-lingual text");
    return out;
}

CosyVoice3TextTokens CosyVoice3TextTokenizer::encode_instruct(
    std::string_view text,
    std::string_view instruction) const {
    CosyVoice3TextTokens out;
    out.prompt = impl_->tokenizer->encode(instruction_with_boundary(instruction), true);
    out.target = impl_->tokenizer->encode(std::string(text), true);
    require_end_of_prompt(out.prompt, "instruction");
    return out;
}

}  // namespace engine::models::cosyvoice3
