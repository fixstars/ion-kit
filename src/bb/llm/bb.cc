#include <fstream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <Halide.h>

#include <llama.h>

#include "ion/export.h"

#include "log.h"
#include "json/json.hpp"

// #include "base64.hpp"
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

}  // namespace llm
}  // namespace bb
}  // namespace ion

#define ION_REGISTER_EXTERN(NAME) static auto ion_register_extern_##NAME = ion::bb::llm::RegisterExtern(#NAME, NAME);

std::string escape_escape_sequences(const std::string &str_) {
    auto str = str_;
    ;
    std::pair<char, char> const sequences[]{
        {'\a', 'a'},
        {'\b', 'b'},
        {'\f', 'f'},
        {'\n', 'n'},
        {'\r', 'r'},
        {'\t', 't'},
        {'\v', 'v'},
    };

    for (size_t i = 0; i < str.length(); ++i) {
        char *const c = str.data() + i;

        for (auto const seq : sequences) {
            if (*c == seq.first) {
                *c = seq.second;
                str.insert(i, "\\");
                ++i;  // to account for inserted "\\"
                break;
            }
        }
    }

    return str;
}
//
// NOTE: Originally defined in llama.cpp
//
struct llava_context {
    struct clip_ctx *ctx_clip = NULL;
    struct llama_context *ctx_llama = NULL;
    struct llama_model *model = NULL;
};

struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens, int n_batch, int *n_past) {
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            ion::log::error("Failed to eval. token {}/{} (batch size {}, n_past {})", i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context *ctx_llama, const char *str, int n_batch, int *n_past, bool add_bos) {
    std::string str2 = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char *sample(struct llama_sampling_context *ctx_sampling,
                          struct llama_context *ctx_llama,
                          int *n_past) {
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

struct llava_image_embed *llava_image_embed_make_with_rawbytes(struct clip_ctx *ctx_clip, int n_threads, const std::vector<uint8_t> &buf, int32_t width, int32_t height) {
    clip_image_u8 *img = clip_image_u8_init();
    img->nx = width;
    img->ny = height;
    img->buf.resize(3 * img->nx * img->ny);
    memcpy(img->buf.data(), reinterpret_cast<const char *>(buf.data()), buf.size());

    float *image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        throw std::runtime_error("Failed to embed the image");
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed *)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static struct llava_image_embed *load_image(llava_context *ctx_llava, gpt_params *params, const std::vector<uint8_t> &buf, int32_t width, int32_t height) {

    // load and preprocess the image
    llava_image_embed *embed = NULL;
    embed = llava_image_embed_make_with_rawbytes(ctx_llava->ctx_clip, params->n_threads, buf, width, height);
    if (!embed) {
        throw std::runtime_error("Failed to embed image from rawbytes");
    }

    return embed;
}

static std::string process_prompt(struct llava_context *ctx_llava, struct llava_image_embed *image_embed, gpt_params *params, const std::string &prompt) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<image>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<image>").length());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int)tmp.size(); i++) {
                ion::log::info("{:6d} -> '{}'", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]));
            }
        }
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int)tmp.size(); i++) {
                ion::log::info("{:6d} -> '{}'", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]));
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int)tmp.size(); i++) {
                ion::log::info("{:6d} -> '{}'", tmp[i], llama_token_to_piece(ctx_llava->ctx_llama, tmp[i]));
            }
        }
    }

    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, true);
    llava_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response
    struct llama_sampling_context *ctx_sampling = llama_sampling_init(params->sparams);
    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char *tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break;                        // Yi-VL behavior
        if (strstr(response.c_str(), "<|im_end|>")) break;    // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break;  // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break;         // mistral llava-1.6
    }

    ion::log::debug("system_prompt:{} user_prompt:{} response:{}", system_prompt, user_prompt, escape_escape_sequences(response));

    llama_sampling_free(ctx_sampling);

    return response;
}

static struct llava_context *llava_init(gpt_params *params) {
    llama_log_set(nullptr, nullptr);

    const char *clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/1);

    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model *model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        ion::log::error("Unable to load model");
        return NULL;
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx = params->n_ctx < 2048 ? 2048 : params->n_ctx;  // we need a longer context size to process image embeddings

    llama_context *ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        ion::log::error("Failed to create the llama_context");
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_new_context_llama(llava_context *ctx_llava, gpt_params *params) {
    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    ctx_params.n_ctx = params->n_ctx < 2048 ? 2048 : params->n_ctx;  // we need a longer context size to process image embeddings

    llama_context *ctx_llama = llama_new_context_with_model(ctx_llava->model, ctx_params);

    if (ctx_llama == NULL) {
        throw std::runtime_error("Failed to create the llama_context");
    }

    if (ctx_llava->ctx_llama != NULL) {
        llama_free(ctx_llava->ctx_llama);
    }
    ctx_llava->ctx_llama = ctx_llama;
}

static llava_context *llava_init_without_ctx_llama(gpt_params *params) {
    const char *clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/1);

    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model *model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        ion::log::error("Unable to load model");
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = nullptr;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;

    return ctx_llava;
}

static void llava_free(struct llava_context *ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

namespace ion {
namespace bb {
namespace llm {
namespace rt {

class Llava {
public:
    static Llava &get_instance() {
        static Llava llava;
        return llava;
    }

    static void release_instance(const std::string &id) {
        auto &llava = get_instance();
        llava.keep_running_ = false;
        llava.thread_.join();
        llava_free(llava.ctx_llava_);
    }

    Llava(const Llava &) = delete;

    ~Llava() {
    }

    bool is_initialized() {
        return initialized_;
    }

    void init(int32_t width, int32_t height) {
        width_ = width;
        height_ = height;

        thread_ = std::thread(entry_point, this);

        ctx_llava_ = llava_init(&params_);
        if (ctx_llava_ == NULL) {
            throw std::runtime_error("Failed to init llava");
        }

        initialized_ = true;
    }

    std::string process(const std::vector<uint8_t> &buf, const std::string &prompt) {

        auto image_embed = load_image(ctx_llava_, &params_, buf, width_, height_);

        // process the prompt
        auto response = process_prompt(ctx_llava_, image_embed, &params_, prompt);

        llava_image_embed_free(image_embed);

        response = response.substr(response.find_last_of('\n') + 1);
        response = response.substr(0, response.find_last_of("</s>") - 4);

        llama_kv_cache_clear(ctx_llava_->ctx_llama);

        return response;
    }

    void post(const Halide::Runtime::Buffer<uint8_t> &ibuf, const std::string &prompt) {
        std::unique_lock<std::mutex> lock(mutex_);

        // clear old
        while (task_queue_.size()) {
            task_queue_.pop();
        }

        task_queue_.emplace(std::make_shared<std::vector<uint8_t>>(ibuf.data(), ibuf.data() + ibuf.size_in_bytes()), prompt);
        cv_.notify_one();
    }

    std::string retrieve() {
        std::unique_lock<std::mutex> lock(mutex_);
        return response_;
    }

private:
    Llava()
        : keep_running_(true), initialized_(false) {
        params_.model = "ggml-mistral-q_4_k.gguf";
        params_.mmproj = "mmproj-mistral7b-f16-q6_k.gguf";
        // params_.model = "llava-phi-3-mini-gguf/ggml-model-int4.gguf";
        // params_.mmproj = "llava-phi-3-mini-gguf/mmproj-model-f16.gguf";
        params_.n_gpu_layers = 999;
        params_.n_ctx = 4096;
    }

    void thread_main() {
        while (keep_running_) {
            std::shared_ptr<std::vector<uint8_t>> bufp;
            std::string prompt;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return task_queue_.size(); });

                std::tie(bufp, prompt) = task_queue_.front();
                task_queue_.pop();
            }

            auto response = process(*bufp, prompt);

            {
                std::unique_lock<std::mutex> lock(mutex_);
                response_ = response;
            }
        }
    }

    static void entry_point(Llava *obj) {
        try {
            obj->thread_main();
        } catch (const std::exception &e) {
            ::std::unique_lock<::std::mutex> lock(obj->mutex_);
            ion::log::error(e.what());
            obj->ep_ = ::std::current_exception();
        }
    }

    gpt_params params_;
    llava_context *ctx_llava_;

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::exception_ptr ep_;
    std::queue<std::tuple<std::shared_ptr<std::vector<uint8_t>>, std::string>> task_queue_;
    std::atomic<bool> keep_running_;

    bool initialized_;
    int32_t width_;
    int32_t height_;

    std::string response_;
};

}  // namespace rt
}  // namespace llm
}  // namespace bb
}  // namespace ion

extern "C" int ION_EXPORT ion_bb_llm_llava_dispose(const char *id) {
    ion::bb::llm::rt::Llava::release_instance(id);
    return 0;
}

extern "C" ION_EXPORT int ion_bb_llm_llava(halide_buffer_t *in, halide_buffer_t *prompt, int32_t width, int32_t height, halide_buffer_t *out) {
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
                prompt->dim[0].extent = 1024;  // TBD
            }

            return 0;
        }

        Halide::Runtime::Buffer<uint8_t> ibuf(*in);
        Halide::Runtime::Buffer<int8_t> pbuf(*prompt);
        Halide::Runtime::Buffer<int8_t> obuf(*out);

        auto &llava = ion::bb::llm::rt::Llava::get_instance();
        if (!llava.is_initialized()) {
            llava.init(width, height);
        }
        // auto response = llava.process(ibuf, std::string(reinterpret_cast<const char*>(pbuf.data())));
        llava.post(ibuf, std::string(reinterpret_cast<const char *>(pbuf.data())));
        auto response = llava.retrieve();

        obuf.fill(0);
        std::memcpy(obuf.data(), response.c_str(), std::min(obuf.size_in_bytes(), response.size()));

        return 0;

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error" << std::endl;
        return 1;
    }
}
ION_REGISTER_EXTERN(ion_bb_llm_llava)

#undef ION_REGISTER_EXTERN
