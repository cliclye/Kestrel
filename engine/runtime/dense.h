/* Dense (Qwen2 / Llama / Mistral) path shared by kestrel-engine. */
#ifndef KESTREL_DENSE_H
#define KESTREL_DENSE_H

/* 1 if SNAP/config.json looks like a dense GQA causal LM (not glm MoE). */
int dense_is_arch(const char *snap);

/* Run dense generate using SNAP / PROMPT|COLI_PROMPT / NGEN env (same as MoE CLI). */
int dense_run(int argc, char **argv);

#endif
