#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const char * prompt = argc > 1 ? argv[1] : "Once upon a time";
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    llama_model * model = llama_model_load_from_file("models/tinyllama-15M-stories-Q3_K_M.gguf", llama_model_default_params());
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 0;  // 0 = the model's trained context length; generation stops when it's full
    llama_context * ctx = llama_init_from_model(model, cparams);
    llama_sampler * smpl = llama_sampler_init_greedy();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(256);
    tokens.resize(llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true));
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    printf("%s", prompt);
    for (llama_token tok; llama_decode(ctx, batch) == 0; batch = llama_batch_get_one(&tok, 1)) {
        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char buf[128];
        fwrite(buf, 1, llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true), stdout);
        fflush(stdout);
    }
    printf("\n");
}
