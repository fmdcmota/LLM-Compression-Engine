The idea for this repository is to make a compression engine for binary text in C++ to improve coding skills in C++. 

**NO AI** will be used - the objective is not a product, but the learning process

This idea was inspired from 3b1b's video *information theory (part 3)*

---

The objective is to achieve the highest compression - computational expense is negligible. We want to approximate __Claude Shannon's Source Coding Theorem__ from the 1948 paper *"A Mathematical Theory of Communication"*. In his 1951 experiments he estimated the long-range entropy of the english language to be 0.6-1.3 bits/character - therefore the long-term goal of this project is to build an algorithm which falls within this.

The current best algorithm (1st winner of the Huffer Prize) is Vladimir Ivanov's fx2-cmix-transformer algorithm - achieving 0.776 bpc on 1GB of wikipedia content.

> Important to note that wikipedia text includes XML markup which may increase entropy compared to Shannon's 1951 experimental estimates

TODO:
 - Write base architecture
 - Find LLM to base model on

Notes
---

Quantization - cuts memory usage with minimal quality loss
 - This will be necessary, as we want to run larger parameter models in order to improve compression [source] without making the compressing software extremely slow. Initially the model will have to run on my AMD Ryzen 9HX 370

llama.cpp will be used to run the 

 - We will contain our model to use .gguf 
