/* Kestrel dense engine — Qwen2 / Llama-style GQA + SwiGLU.
 *
 * Same product binary family as kestrel-engine (glm_moe_dsa). Detected via
 * config.json model_type / architectures. Weights loaded from HF safetensors,
 * quantized to int8(+per-row scale) on load so 1.5B–3B class models fit a
 * 16GB Mac; matmul uses ARM NEON IDOT when available (same idea as MoE path).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "st.h"
#include "tok.h"
#include "dense.h"

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab;
    float theta, eps;
    int eos_id, bos_id;
    int tie_emb; /* lm_head == embed */
} DCfg;

typedef struct {
    float *in_ln, *post_ln;
    int8_t *q, *k, *v, *o, *gate, *up, *down;
    float *qs, *ks, *vs, *os, *gates, *ups, *downs;
    float *qb, *kb, *vb; /* optional q/k/v bias (Qwen2) */
} DLayer;

typedef struct {
    DCfg c;
    shards S;
    float *embed, *lm_head, *final_norm;
    DLayer *L;
    float **K, **V;
    int kv_len, max_t;
    double load_s;
    /* reusable scratch (avoids malloc/free per layer/token) */
    float *ws_x, *ws_nrm, *ws_tmp;
    float *ws_q, *ws_k, *ws_v, *ws_ctx, *ws_sc;
    float *ws_g, *ws_u, *ws_logit;
    float *rope_inv; /* head_dim/2 inv frequencies */
    int8_t *idot_xi;
    float *idot_xs;
    int idot_cap_i;
} DModel;

static DModel *g_dens = NULL; /* active model for matmul scratch */

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#if defined(__APPLE__)
static double rss_gb(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
}
#else
static double rss_gb(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss / (1024.0 * 1024.0);
}
#endif

static float *falloc(int64_t n) {
    float *p = calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}

static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I) {
    const int qmax = 127;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f;
        for (int i = 0; i < I; i++) {
            float a = fabsf(wr[i]);
            if (a > amax) amax = a;
        }
        float s = amax / (float)qmax;
        if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v > qmax) v = qmax;
            if (v < -qmax - 1) v = -qmax - 1;
            qr[i] = (int8_t)v;
        }
    }
}

#if defined(__ARM_NEON)
static inline int32_t dens_dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif

/* y[O] = x[I] @ W^T  with W as int8 + per-row scale */
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    static int idot = -1;
    if (idot < 0) {
        const char *e = getenv("IDOT");
        idot = !(e && *e == '0');
    }
    if (idot && (I % 16) == 0 && I <= 16384 && g_dens) {
        int nb = I / 16;
        if (g_dens->idot_cap_i < I) {
            free(g_dens->idot_xi);
            free(g_dens->idot_xs);
            g_dens->idot_xi = (int8_t *)malloc((size_t)I);
            g_dens->idot_xs = (float *)malloc((size_t)nb * sizeof(float));
            if (!g_dens->idot_xi || !g_dens->idot_xs) {
                fprintf(stderr, "OOM idot scratch\n");
                exit(1);
            }
            g_dens->idot_cap_i = I;
        }
        int8_t *xi = g_dens->idot_xi;
        float *xs = g_dens->idot_xs;
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b * 16;
            float am = 0.f;
            for (int i = 0; i < 16; i++) {
                float a = fabsf(xb[i]);
                if (a > am) am = a;
            }
            float s = am / 127.f;
            if (s < 1e-12f) s = 1e-12f;
            xs[b] = s;
            float inv = 1.f / s;
            for (int i = 0; i < 16; i++) xi[b * 16 + i] = (int8_t)lrintf(xb[i] * inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++)
                acc += xs[b] * (float)dens_dot_i8_16(xi + b * 16, w + b * 16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0;
    for (int i = 0; i < D; i++) ms += (double)x[i] * x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static void rope_head(float *x, int pos, int head_dim, const float *inv_freq) {
    int h = head_dim / 2;
    for (int j = 0; j < h; j++) {
        float ang = (float)pos * inv_freq[j], cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j + h];
        x[j] = a * cs - b * sn;
        x[j + h] = b * cs + a * sn;
    }
}

static int gi(jval *r, const char *k, int def) {
    jval *v = json_get(r, k);
    return v ? (int)v->num : def;
}

static void dens_load_cfg(DCfg *c, const char *snap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { /* ignore */ }
    buf[n] = 0;
    fclose(f);
    char *arena = NULL;
    jval *r = json_parse(buf, &arena);
    c->hidden = gi(r, "hidden_size", 0);
    c->n_layers = gi(r, "num_hidden_layers", 0);
    c->n_heads = gi(r, "num_attention_heads", 0);
    c->n_kv_heads = gi(r, "num_key_value_heads", c->n_heads);
    c->inter = gi(r, "intermediate_size", 0);
    c->vocab = gi(r, "vocab_size", 0);
    c->head_dim = c->n_heads ? (c->hidden / c->n_heads) : 0;
    jval *th = json_get(r, "rope_theta");
    c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r, "rms_norm_eps");
    c->eps = ep ? (float)ep->num : 1e-6f;
    jval *eos = json_get(r, "eos_token_id");
    c->eos_id = eos ? (int)eos->num : -1;
    jval *bos = json_get(r, "bos_token_id");
    c->bos_id = bos ? (int)bos->num : -1;
    jval *tie = json_get(r, "tie_word_embeddings");
    c->tie_emb = (tie && tie->t == J_BOOL) ? tie->boolean : 0;
    if (c->hidden <= 0 || c->n_layers <= 0 || c->n_heads <= 0 || c->inter <= 0 || c->vocab <= 0) {
        fprintf(stderr, "dense: invalid config in %s\n", path);
        exit(1);
    }
    free(buf);
    free(arena);
}

static float *load_f32(DModel *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) return NULL;
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

static void load_qweight(DModel *m, const char *name, int8_t **q, float **scale, int O, int I) {
    float *tmp = load_f32(m, name);
    if (!tmp) { fprintf(stderr, "dense: missing %s\n", name); exit(1); }
    *q = (int8_t *)malloc((size_t)O * (size_t)I);
    *scale = falloc(O);
    if (!*q) { fprintf(stderr, "OOM quant %s\n", name); exit(1); }
    quantize_rows(tmp, *q, *scale, O, I);
    free(tmp);
}

static void dens_model_init(DModel *m, const char *snap) {
    memset(m, 0, sizeof(*m));
    dens_load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    DCfg *c = &m->c;
    double t0 = now_s();
    m->embed = load_f32(m, "model.embed_tokens.weight");
    if (!m->embed) { fprintf(stderr, "dense: missing embed_tokens\n"); exit(1); }
    m->final_norm = load_f32(m, "model.norm.weight");
    if (!m->final_norm) { fprintf(stderr, "dense: missing model.norm\n"); exit(1); }
    m->lm_head = load_f32(m, "lm_head.weight");
    if (!m->lm_head) {
        if (c->tie_emb) m->lm_head = m->embed;
        else { fprintf(stderr, "dense: missing lm_head and tie_word_embeddings=false\n"); exit(1); }
    }
    m->L = calloc((size_t)c->n_layers, sizeof(DLayer));
    char nm[320];
    int D = c->hidden, I = c->inter, H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim;
    for (int i = 0; i < c->n_layers; i++) {
        DLayer *l = &m->L[i];
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", i);
        l->in_ln = load_f32(m, nm);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", i);
        l->post_ln = load_f32(m, nm);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", i);
        load_qweight(m, nm, &l->q, &l->qs, H * hd, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", i);
        load_qweight(m, nm, &l->k, &l->ks, KV * hd, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", i);
        load_qweight(m, nm, &l->v, &l->vs, KV * hd, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.bias", i);
        l->qb = load_f32(m, nm);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.bias", i);
        l->kb = load_f32(m, nm);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.bias", i);
        l->vb = load_f32(m, nm);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", i);
        load_qweight(m, nm, &l->o, &l->os, D, H * hd);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", i);
        load_qweight(m, nm, &l->gate, &l->gates, I, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", i);
        load_qweight(m, nm, &l->up, &l->ups, I, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", i);
        load_qweight(m, nm, &l->down, &l->downs, D, I);
    }
    m->load_s = now_s() - t0;
    /* rope inv freq + forward scratch */
    m->rope_inv = falloc(hd / 2);
    for (int j = 0; j < hd / 2; j++)
        m->rope_inv[j] = powf(c->theta, -2.0f * j / (float)hd);
    m->ws_x = falloc(D);
    m->ws_nrm = falloc(D);
    m->ws_tmp = falloc(D);
    m->ws_q = falloc(H * hd);
    m->ws_k = falloc(KV * hd);
    m->ws_v = falloc(KV * hd);
    m->ws_ctx = falloc(H * hd);
    m->ws_g = falloc(I);
    m->ws_u = falloc(I);
    m->ws_logit = falloc(c->vocab);
    m->ws_sc = NULL; /* sized with max_t later */
    m->idot_xi = NULL;
    m->idot_xs = NULL;
    m->idot_cap_i = 0;
    fprintf(stderr, "[dense] loaded %s in %.1fs | RSS %.2f GB | layers=%d hidden=%d\n",
            snap, m->load_s, rss_gb(), c->n_layers, c->hidden);
}

static void dens_attention(DModel *m, DLayer *l, int layer, float *x, int pos, float *out) {
    DCfg *c = &m->c;
    int D = c->hidden, H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim;
    int gqa = H / KV;
    float *q = m->ws_q, *k = m->ws_k, *v = m->ws_v;
    matmul_q(q, x, l->q, l->qs, D, H * hd);
    matmul_q(k, x, l->k, l->ks, D, KV * hd);
    matmul_q(v, x, l->v, l->vs, D, KV * hd);
    if (l->qb) for (int i = 0; i < H * hd; i++) q[i] += l->qb[i];
    if (l->kb) for (int i = 0; i < KV * hd; i++) k[i] += l->kb[i];
    if (l->vb) for (int i = 0; i < KV * hd; i++) v[i] += l->vb[i];
    for (int hh = 0; hh < H; hh++) rope_head(q + hh * hd, pos, hd, m->rope_inv);
    for (int hh = 0; hh < KV; hh++) rope_head(k + hh * hd, pos, hd, m->rope_inv);
    for (int hh = 0; hh < KV; hh++) {
        memcpy(m->K[layer] + ((int64_t)hh * m->max_t + pos) * hd, k + hh * hd, (size_t)hd * sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh * m->max_t + pos) * hd, v + hh * hd, (size_t)hd * sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = m->ws_ctx;
    float *sc = m->ws_sc;
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / gqa;
        const float *qv = q + hh * hd;
        for (int t = 0; t <= pos; t++) {
            const float *kv = m->K[layer] + ((int64_t)kvh * m->max_t + t) * hd;
            float acc = 0.f;
            for (int d = 0; d < hd; d++) acc += qv[d] * kv[d];
            sc[t] = acc * scale;
        }
        softmax_row(sc, pos + 1);
        float *cx = ctx + hh * hd;
        for (int d = 0; d < hd; d++) cx[d] = 0.f;
        for (int t = 0; t <= pos; t++) {
            const float *vr = m->V[layer] + ((int64_t)kvh * m->max_t + t) * hd;
            float a = sc[t];
            for (int d = 0; d < hd; d++) cx[d] += a * vr[d];
        }
    }
    matmul_q(out, ctx, l->o, l->os, H * hd, D);
}

static void dens_mlp(DModel *m, DLayer *l, const float *x, float *out) {
    int D = m->c.hidden, I = m->c.inter;
    float *g = m->ws_g, *u = m->ws_u;
    matmul_q(g, x, l->gate, l->gates, D, I);
    matmul_q(u, x, l->up, l->ups, D, I);
    for (int i = 0; i < I; i++) {
        float gv = g[i];
        g[i] = (gv / (1.f + expf(-gv))) * u[i]; /* silu(gate)*up */
    }
    matmul_q(out, g, l->down, l->downs, I, D);
}

static float *dens_step(DModel *m, int token, int pos) {
    DCfg *c = &m->c;
    int D = c->hidden;
    float *x = m->ws_x, *nrm = m->ws_nrm, *tmp = m->ws_tmp;
    memcpy(x, m->embed + (int64_t)token * D, (size_t)D * sizeof(float));
    for (int i = 0; i < c->n_layers; i++) {
        DLayer *l = &m->L[i];
        rmsnorm_row(nrm, x, l->in_ln, D, c->eps);
        dens_attention(m, l, i, nrm, pos, tmp);
        for (int d = 0; d < D; d++) x[d] += tmp[d];
        rmsnorm_row(nrm, x, l->post_ln, D, c->eps);
        dens_mlp(m, l, nrm, tmp);
        for (int d = 0; d < D; d++) x[d] += tmp[d];
    }
    m->kv_len = pos + 1;
    rmsnorm_row(nrm, x, m->final_norm, D, c->eps);
    float *logit = m->ws_logit;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < c->vocab; o++) {
        const float *w = m->lm_head + (int64_t)o * D;
        float acc = 0.f;
        for (int i = 0; i < D; i++) acc += nrm[i] * w[i];
        logit[o] = acc;
    }
    return logit;
}

static int argmax(const float *x, int n) {
    int b = 0;
    float v = x[0];
    for (int i = 1; i < n; i++) if (x[i] > v) { v = x[i]; b = i; }
    return b;
}

int dense_is_arch(const char *snap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { /* ignore */ }
    buf[n] = 0;
    fclose(f);
    int hit = 0;
    if (strstr(buf, "\"qwen2\"") || strstr(buf, "Qwen2ForCausalLM") ||
        strstr(buf, "\"llama\"") || strstr(buf, "LlamaForCausalLM") ||
        strstr(buf, "\"mistral\"") || strstr(buf, "MistralForCausalLM"))
        hit = 1;
    if (strstr(buf, "glm_moe_dsa") || strstr(buf, "n_routed_experts"))
        hit = 0;
    free(buf);
    return hit;
}

int dense_run(int argc, char **argv) {
    (void)argc;
    (void)argv;
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "SNAP=<dir>\n"); return 1; }
    const char *prompt = getenv("COLI_PROMPT");
    if (!prompt) prompt = getenv("PROMPT");
    if (!prompt) prompt = "Say hello in one short sentence.";
    int ngen = getenv("NGEN") ? atoi(getenv("NGEN")) : 32;
    if (ngen < 1) ngen = 1;
    if (ngen > 512) ngen = 512;
    int quiet = getenv("QUIET") ? atoi(getenv("QUIET")) : 0;

    DModel m;
    dens_model_init(&m, snap);
    g_dens = &m;
    DCfg *c = &m.c;

    char tkp[2048];
    snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    Tok T;
    tok_load(&T, tkp);
    int eos = tok_id_of(&T, "<|im_end|>");
    if (eos < 0) eos = tok_id_of(&T, "<|endoftext|>");
    if (eos < 0) eos = c->eos_id;

    int cap = (int)strlen(prompt) + 64;
    if (cap < 256) cap = 256;
    int *prompt_ids = malloc((size_t)cap * sizeof(int));
    if (!prompt_ids) { fprintf(stderr, "OOM prompt ids\n"); return 1; }
    int np = tok_encode(&T, prompt, (int)strlen(prompt), prompt_ids, cap);
    if (np < 1) {
        fprintf(stderr, "dense: prompt empty after tokenization\n");
        return 1;
    }

    m.max_t = np + ngen + 8;
    m.ws_sc = falloc(m.max_t);
    m.K = calloc((size_t)c->n_layers, sizeof(float *));
    m.V = calloc((size_t)c->n_layers, sizeof(float *));
    for (int i = 0; i < c->n_layers; i++) {
        m.K[i] = falloc((int64_t)c->n_kv_heads * m.max_t * c->head_dim);
        m.V[i] = falloc((int64_t)c->n_kv_heads * m.max_t * c->head_dim);
    }

    fprintf(stderr, "[dense] prefill %d tokens, generate up to %d (eos=%d)\n", np, ngen, eos);
    double t_pre = now_s();
    float *logit = NULL;
    for (int i = 0; i < np; i++)
        logit = dens_step(&m, prompt_ids[i], i);
    double prefill_s = now_s() - t_pre;
    if (!quiet)
        fprintf(stderr, "[dense] prefill %.2fs (%.2f tok/s)\n",
                prefill_s, prefill_s > 0 ? np / prefill_s : 0);

    double t0 = now_s();
    int generated = 0;
    char outbuf[4096];
    for (int s = 0; s < ngen; s++) {
        int tok = argmax(logit, c->vocab);
        generated++;
        int nch = tok_decode(&T, &tok, 1, outbuf, (int)sizeof(outbuf) - 1);
        if (nch > 0) {
            outbuf[nch] = 0;
            fputs(outbuf, stdout);
            fflush(stdout);
        }
        if (eos >= 0 && tok == eos) break;
        if (c->eos_id >= 0 && tok == c->eos_id) break;
        if (s + 1 == ngen) break;
        logit = dens_step(&m, tok, np + s);
    }
    fputc('\n', stdout);
    fflush(stdout);
    double dt = now_s() - t0;
    double tps = dt > 0 ? (generated / dt) : 0;
    fprintf(stderr, "[dense] decode %.2f tok/s (%.2fs for %d toks) | RSS %.2f GB | load %.1fs\n",
            tps, dt, generated, rss_gb(), m.load_s);
    free(prompt_ids);
    g_dens = NULL;
    return 0;
}
