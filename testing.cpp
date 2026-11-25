// Flags you can pass:
//   -m <path>    which model to load   (defaults to the tiny stories model)
//   -n <int>     how many tokens to generate   (128)
//   -t <float>   temperature; leave it at 0 to always take the most likely token
//   -s <int>     seed, so a sampled run can be reproduced   (random if you don't set it)
// Run it with no prompt and it drops into interactive mode.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct Options {
    std::string model_path = "models/Qwen3.8-9B-Q4_K_M.gguf";
    std::string prompt;
    int         n_predict  = 128;
    float       temp       = 0.0f;
    uint32_t    seed       = LLAMA_DEFAULT_SEED;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s [-m model.gguf] [-n n_predict] [-t temp] [-s seed] [prompt]\n"
        "       no prompt -> interactive mode\n", argv0);
}

static bool parse_args(int argc, char ** argv, Options & opt) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        if      (a == "-m") { auto v = next("-m"); if (!v) return false; opt.model_path = v; }
        else if (a == "-n") { auto v = next("-n"); if (!v) return false; opt.n_predict  = std::atoi(v); }
        else if (a == "-t") { auto v = next("-t"); if (!v) return false; opt.temp       = std::atof(v); }
        else if (a == "-s") { auto v = next("-s"); if (!v) return false; opt.seed       = std::strtoul(v, nullptr, 10); }
        else if (a == "-h" || a == "--help") { return false; }
        else {
            // Anything that isn't a flag is part of the prompt, so quotes are optional
            if (!opt.prompt.empty()) opt.prompt += ' ';
            opt.prompt += a;
        }
    }
    return true;
}

// llama.cpp is very chatty while loading, so only let real errors through.
// CONT lines just continue the previous message, so treat them the same way it was treated.
static void quiet_log(ggml_log_level level, const char * text, void *) {
    static ggml_log_level last = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (last == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

static llama_sampler * make_sampler(const Options & opt) {
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (opt.temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        // Keep the 40 most likely tokens, trim to the top 95% of probability,
        // apply the temperature, then pick one at random
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(opt.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(opt.seed));
    }
    return smpl;
}

// Feeds in the prompt and prints whatever the model writes after it, token by token.
static bool generate(llama_context * ctx, const llama_vocab * vocab, llama_sampler * smpl,
                     const std::string & prompt, int n_predict) {
    // Forget the previous prompt so every run starts clean
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_sampler_reset(smpl);

    const int n_ctx = llama_n_ctx(ctx);

    // First call just tells us how many tokens we need, the second actually fills them in
    int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
        std::fprintf(stderr, "tokenize failed\n");
        return false;
    }
    if (n_prompt >= n_ctx) {
        std::fprintf(stderr, "prompt is %d tokens, but the context only holds %d\n", n_prompt, n_ctx);
        return false;
    }

    std::printf("%s", prompt.c_str());
    std::fflush(stdout);

    // The first batch is the whole prompt; after that we feed back one token at a time
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    int n_pos = 0;
    for (int i = 0; i < n_predict; ++i) {
        if (n_pos + batch.n_tokens >= n_ctx) {
            std::printf("\n[context full]");
            break;
        }
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "\ndecode failed\n");
            return false;
        }
        n_pos += batch.n_tokens;

        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;  // the model decided it's done

        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::fprintf(stderr, "\ntoken_to_piece failed\n");
            return false;
        }
        std::fwrite(buf, 1, n, stdout);
        std::fflush(stdout);  // print each token as soon as it arrives

        batch = llama_batch_get_one(&tok, 1);
    }
    std::printf("\n");
    std::fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    Options opt;

    if (!parse_args(argc, argv, opt)) {
        usage(argv[0]);
        return 1;
    }

    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = [](float, void *) { return true; };  // no row of dots while loading
    llama_model * model = llama_model_load_from_file(opt.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", opt.model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Small models are trained on short contexts (this one only 256 tokens), and they
    // turn to gibberish past that, so don't give them more room than they were trained on.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = llama_model_n_ctx_train(model);
    cparams.n_batch = cparams.n_ctx;  // big enough to take the whole prompt in one go
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_sampler * smpl = make_sampler(opt);

    int rc = 0;
    if (!opt.prompt.empty()) {
        rc = generate(ctx, vocab, smpl, opt.prompt, opt.n_predict) ? 0 : 1;
    } else {
        std::printf("model: %s (ctx %d)\nType a prompt and press Enter. Empty line or Ctrl-D quits.\n",
                    opt.model_path.c_str(), llama_n_ctx(ctx));
        std::string line;
        while (true) {
            std::printf("\n> ");
            std::fflush(stdout);
            if (!std::getline(std::cin, line) || line.empty()) break;
            generate(ctx, vocab, smpl, line, opt.n_predict);
        }
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return rc;
}