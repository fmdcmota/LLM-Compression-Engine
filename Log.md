24/11/2026

Got the time to play around with model and made testing.cpp to vary temperature, token length, etc.

Further steps will be to figure out how to extract ranks and encode sentences into fp numbers

Note that I will probably have to switch to a larger model - TinyStories is worse than I imagined - looking at qwen using llama

Will stick with llama as it gives fine-grained control over GGUF quantizations, GPU layer offloading, and backends (such as CUDA)
*Want to explore this further when I implement the new model.