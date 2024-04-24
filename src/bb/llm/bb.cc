#include <fstream>

#include <Halide.h>

#include <llama.h>

#include "ion/export.h"

#include "log.h"
#include "json/json.hpp"

#include "base64.hpp"
#include "clip.h"
#include "common.h"
#include "sampling.h"
#include "llava.h"

namespace ion {
namespace bb {
namespace llm {

std::map<std::string, Halide::ExternCFunction> extern_functions;

class RegisterExtern {
 public:
     RegisterExtern(std::string key, Halide::ExternCFunction f) {
         extern_functions[key] = f;
     }
};

} // llm
} // bb
} // ion

#define ION_REGISTER_EXTERN(NAME) static auto ion_register_extern_##NAME = ion::bb::llm::RegisterExtern(#NAME, NAME);

// NOTE: Originally defined in llama.cpp/clip.cpp
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            fprintf(stderr, "%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (id == llama_token_eos(llama_get_model(ctx_llama))) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static llava_image_embed * llava_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        fprintf(stderr, "%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        fprintf(stderr, "%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

struct llava_image_embed * llava_image_embed_make_with_rawbytes(struct clip_ctx * ctx_clip, int n_threads, Halide::Runtime::Buffer<uint8_t> buf) {
    clip_image_u8 * img = clip_image_u8_init();
    img->nx = buf.dim(1).extent();
    img->ny = buf.dim(2).extent();
    img->buf.resize(3 * img->nx * img->ny);
    memcpy(img->buf.data(), reinterpret_cast<const char*>(buf.data()), buf.size_in_bytes());

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static struct llava_image_embed * load_image_(llava_context * ctx_llava, gpt_params *params, const Halide::Runtime::Buffer<uint8_t>& ibuf) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    embed = llava_image_embed_make_with_rawbytes(ctx_llava->ctx_clip, params->n_threads, ibuf);
    if (!embed) {
        fprintf(stderr, "%s: is %s really an image file?\n", __func__, params->image.c_str());
        return NULL;
    }

    return embed;
}
static struct llava_image_embed * load_image(llava_context * ctx_llava, gpt_params * params) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            fprintf(stderr, "using base64 encoded image instead of command line image path\n");
        }
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->n_threads, prompt);
        if (!embed) {
            fprintf(stderr, "%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->n_threads, params->image.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, params->image.c_str());
            return NULL;
        }
    }

    return embed;
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, gpt_params * params, const std::string & prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<image>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<image>").length());
        printf("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        printf("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response

    fprintf(stderr, "\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        printf("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    llama_sampling_free(ctx_sampling);
    printf("\n");
}


static struct llava_context * llava_init(gpt_params * params) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return NULL;
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

extern "C"
ION_EXPORT int ion_bb_llm_llava(halide_buffer_t *in, halide_buffer_t *prompt, int32_t width, int32_t height, halide_buffer_t *out) {
    try {
        if (in->is_bounds_query() || prompt->is_bounds_query()) {
            if (in->is_bounds_query()) {
                in->dim[0].min = 0;
                in->dim[0].extent = 3;
                in->dim[1].min = 0;
                in->dim[1].extent = width;
                in->dim[2].min = 0;
                in->dim[2].extent = height;
            }
            
            if (prompt->is_bounds_query()) {
                prompt->dim[0].min = 0;
                prompt->dim[0].extent = 1024; // TBD
            }

            return 0;
        }

        Halide::Runtime::Buffer<uint8_t> ibuf(*in);
        Halide::Runtime::Buffer<int8_t> pbuf(*prompt);
        Halide::Runtime::Buffer<int8_t> obuf(*out);

        // std::ofstream ofs("test.bin");
        // ofs.write(reinterpret_cast<const char*>(ibuf.data()), in->size_in_bytes());
#if 0        
        // 
        // Initialization
        // 
        llama_backend_init();
        // llama_numa_init(params->numa);

        // Load CLIP model
        auto verbosity = 0;
        auto ctx_clip = clip_model_load("/home/iitaku/Develop/llava-1.6-gguf/mmproj-mistral7b-f16-q6_k.gguf", verbosity);

        // Load LLM model 
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999; // TODO

        llama_model *model = llama_load_model_from_file("/home/iitaku/Develop/llava-1.6-gguf/ggml-mistral-q_4_k.gguf", model_params);
        if (model == NULL) {
            ion::log::error("Failed to load model");
            return 1;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 4096;  // TODO

        llama_context *ctx_llama = llama_new_context_with_model(model, ctx_params);
        if (ctx_llama == NULL) {
            ion::log::error("Failed to create the llama_context");
            return 1;
        }

        // 
        // Process
        // 
        
        // Run CLIP model
        auto img = clip_image_u8_init();
        img->nx = width;
        img->ny = height;
        img->buf.resize(3 * img->nx * img->ny);
        memcpy(img->buf.data(), ibuf.data(), img->buf.size());

        auto n_threads = 1;
        llava_image_embed img_embed;
        auto embed = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &img_embed.embed, &img_embed.n_image_pos);
        if (!embed) {
            ion::log::error("Failed to load image");
            return 1;
        }

        // Run LLM model
        int n_past = 0;
        auto n_batch = 2048;  // TODO
        std::string user_prompt(reinterpret_cast<const char *>(pbuf.data()));
        llava_eval_image_embed(ctx_llama, &img_embed, n_batch, &n_past);
        eval_string(ctx_llama, user_prompt.c_str(), n_batch, &n_past, false);

        llama_sampling_params sparams;
        struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);
        std::string response = "";
        auto max_tgt_len = 256;
        for (int i = 0; i < max_tgt_len; i++) {
            const char *tmp = sample(ctx_sampling, ctx_llama, &n_past);
            response += tmp;
            if (strcmp(tmp, "</s>") == 0) break;
            if (strstr(tmp, "###")) break;  // Yi-VL behavior
            printf("%s", tmp);
            if (strstr(response.c_str(), "<|im_end|>")) break;    // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
            if (strstr(response.c_str(), "<|im_start|>")) break;  // Yi-34B llava-1.6
            if (strstr(response.c_str(), "USER:")) break;         // mistral llava-1.6

            fflush(stdout);
        }

        llama_sampling_free(ctx_sampling);
#else
        gpt_params params;

        params.model = "/home/iitaku/Develop/llava-1.6-gguf/ggml-mistral-q_4_k.gguf";
        params.mmproj = "/home/iitaku/Develop/llava-1.6-gguf/mmproj-mistral7b-f16-q6_k.gguf";
        params.prompt = "<image>Explain the image in one sentence";
        // params.image = "/home/iitaku/sample1.jpg";
        params.n_gpu_layers = 32;
        params.n_ctx = 4096;

        auto ctx_llava = llava_init(&params);
        if (ctx_llava == NULL) {
            fprintf(stderr, "%s: error: failed to init llava\n", __func__);
            return 1;
        }

        auto image_embed = load_image_(ctx_llava, &params, ibuf);
        //auto image_embed = load_image(ctx_llava, &params);
        if (!image_embed) {
            return 1;
        }

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_print_timings(ctx_llava->ctx_llama);

        llava_image_embed_free(image_embed);
        llava_free(ctx_llava);

#endif
        obuf.fill(0);
        obuf(0) = 'x';

        return 0;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return -1;
    }
}
ION_REGISTER_EXTERN(ion_bb_llm_llava)

#undef ION_REGISTER_EXTERN
