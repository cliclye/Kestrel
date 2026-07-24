/* Kestrel dense engine — Qwen2 / Llama-style GQA + SwiGLU.
 *
 * Speed + RAM: weights kept int8 (incl. tied embed/lm_head); decode uses the
 * same ARM SDOT IDOT family as the MoE path (row-quant activations + 4-acc
 * vdot). Target: beat stock transformers CPU tok/s while staying well under
 * its RSS on 16GB Macs.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#if defined(_WIN32)
#include "compat.h"
#elif defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include "idot_avx.h"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "tok.h"
#include "dense.h"

typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim, inter, vocab;
    float theta, eps;
    int eos_id, bos_id;
    int tie_emb;
} DCfg;

typedef struct {
    float *in_ln, *post_ln;
    int8_t *q, *k, *v;
    float *qs, *ks, *vs;
    float *qb, *kb, *vb;
    float *q_norm, *k_norm; /* Qwen3 optional per-head RMSNorm */
    /* MLP + o_proj in int4 (packed) — decode is bandwidth-bound */
    uint8_t *gate4, *up4, *down4, *o4;
    float *gates, *ups, *downs, *os;
} DLayer;

typedef struct {
    DCfg c;
    shards S;
    int8_t *embed_q;
    float *embed_s;
    int8_t *lm_q;   /* NULL → tied: use embed_q for lookup; logits prefer lm4 */
    float *lm_s;
    uint8_t *lm4;   /* int4 lm_head (or tied embed) for logit matmul */
    float *lm4_s;
    float *final_norm;
    DLayer *L;
    float **K, **V;
    int kv_len, max_t;
    double load_s;
    float *ws_x, *ws_nrm, *ws_tmp;
    float *ws_q, *ws_k, *ws_v, *ws_ctx, *ws_sc;
    float *ws_g, *ws_u, *ws_logit;
    float *rope_inv;
    int8_t *xq; /* activation quant scratch */
    int xq_cap;
    /* optional profile accumulators (DENSE_PROF=1) */
    double t_attn, t_mlp, t_lm, t_other;
    int prof;
} DModel;

static DModel *g_dens = NULL;

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

/* Quantize one activation row → int8; return absmax/127 scale. */
static float qrow_i8(const float *x, int8_t *q, int I) {
    float amax = 0.f;
#if defined(__ARM_NEON)
    int i = 0;
    float32x4_t am = vdupq_n_f32(0.f);
    for (; i + 4 <= I; i += 4) {
        float32x4_t v = vabsq_f32(vld1q_f32(x + i));
        am = vmaxq_f32(am, v);
    }
    amax = vmaxvq_f32(am);
    for (; i < I; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
#else
    for (int i = 0; i < I; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
#endif
    float s = amax / 127.f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.f / s;
    for (int i = 0; i < I; i++) q[i] = (int8_t)lrintf(x[i] * inv);
    return s;
}

/* Engine-class int8·int8 dot (AVX2 on x86, 4-acc SDOT on Apple Silicon). */
static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
#if defined(__AVX2__)
    (void)i;
    return wh_dot_i8i8_avx(w, x, I);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    for (; i + 64 <= I; i += 64) {
        a0 = vdotq_s32(a0, vld1q_s8(w + i), vld1q_s8(x + i));
        a1 = vdotq_s32(a1, vld1q_s8(w + i + 16), vld1q_s8(x + i + 16));
        a2 = vdotq_s32(a2, vld1q_s8(w + i + 32), vld1q_s8(x + i + 32));
        a3 = vdotq_s32(a3, vld1q_s8(w + i + 48), vld1q_s8(x + i + 48));
    }
    int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    for (; i + 16 <= I; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(w + i), vld1q_s8(x + i));
    sum = vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    int32x4_t acc = vdupq_n_s32(0);
    for (; i + 16 <= I; i += 16) {
        int8x16_t wv = vld1q_s8(w + i), xv = vld1q_s8(x + i);
        int16x8_t p = vmull_s8(vget_low_s8(wv), vget_low_s8(xv));
        p = vmlal_s8(p, vget_high_s8(wv), vget_high_s8(xv));
        acc = vpadalq_s16(acc, p);
    }
    sum = vaddvq_s32(acc);
#endif
    for (; i < I; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}

static void ensure_xq(DModel *m, int I) {
    if (m->xq_cap < I) {
        free(m->xq);
        m->xq = (int8_t *)malloc((size_t)I);
        if (!m->xq) { fprintf(stderr, "OOM xq\n"); exit(1); }
        m->xq_cap = I;
    }
}

/* Exact int8 matmul into y; must be called inside an OpenMP parallel region. */
static void matmul_q_exact_rows(float *y, const float *x, const int8_t *q, const float *scale,
                                int I, int O) {
    #pragma omp for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
#if defined(__AVX2__)
        y[o] = wh_dot_f32_i8_avx(x, w, I) * scale[o];
#else
        float acc = 0.f;
        int i = 0;
#if defined(__ARM_NEON)
        float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
        for (; i + 8 <= I; i += 8) {
            int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
            ac0 = vfmaq_f32(ac0, vld1q_f32(x + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
            ac1 = vfmaq_f32(ac1, vld1q_f32(x + i + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
        }
        acc = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
#endif
    }
}

/* y[O] = x[I] @ W^T  (W int8 + per-row scale). IDOT with single act scale. */
static void matmul_q_ex(float *y, const float *x, const int8_t *q, const float *scale,
                        int I, int O, int allow_idot) {
    static int idot = -1;
    if (idot < 0) {
        const char *e = getenv("IDOT");
        idot = !(e && *e == '0');
    }
    if (allow_idot && idot && g_dens) {
        ensure_xq(g_dens, I);
        float sx = qrow_i8(x, g_dens->xq, I);
        const int8_t *xq = g_dens->xq;
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++)
            y[o] = (float)dot_i8i8(q + (int64_t)o * I, xq, I) * scale[o] * sx;
        return;
    }
    /* Exact path: AVX2 / NEON f32×int8 — skip OpenMP on tiny O (k/v projs) to avoid spawn tax. */
    #pragma omp parallel for schedule(static) if(O >= 512)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
#if defined(__AVX2__)
        y[o] = wh_dot_f32_i8_avx(x, w, I) * scale[o];
#else
        float acc = 0.f;
        int i = 0;
#if defined(__ARM_NEON)
        float32x4_t ac0 = vdupq_n_f32(0), ac1 = vdupq_n_f32(0);
        for (; i + 8 <= I; i += 8) {
            int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
            ac0 = vfmaq_f32(ac0, vld1q_f32(x + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))));
            ac1 = vfmaq_f32(ac1, vld1q_f32(x + i + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))));
        }
        acc = vaddvq_f32(vaddq_f32(ac0, ac1));
#endif
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
#endif
    }
}

static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    matmul_q_ex(y, x, q, scale, I, O, 1);
}

/* Fuse gate+up: one activation quant, two weight dots. */
static void matmul_q_pair(float *yg, float *yu, const float *x,
                          const int8_t *qg, const float *sg,
                          const int8_t *qu, const float *su, int I, int O) {
    static int idot = -1;
    if (idot < 0) {
        const char *e = getenv("IDOT");
        idot = !(e && *e == '0');
    }
    if (idot && g_dens) {
        ensure_xq(g_dens, I);
        float sx = qrow_i8(x, g_dens->xq, I);
        const int8_t *xq = g_dens->xq;
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            yg[o] = (float)dot_i8i8(qg + (int64_t)o * I, xq, I) * sg[o] * sx;
            yu[o] = (float)dot_i8i8(qu + (int64_t)o * I, xq, I) * su[o] * sx;
        }
        return;
    }
    matmul_q(yg, x, qg, sg, I, O);
    matmul_q(yu, x, qu, su, I, O);
}

static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    float ms = 0.f;
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.f);
    for (; i + 4 <= D; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    ms = vaddvq_f32(acc);
#endif
    for (; i < D; i++) ms += x[i] * x[i];
    float r = 1.f / sqrtf(ms / (float)D + eps);
    i = 0;
#if defined(__ARM_NEON)
    float32x4_t vr = vdupq_n_f32(r);
    for (; i + 4 <= D; i += 4) {
        float32x4_t xv = vld1q_f32(x + i), wv = vld1q_f32(w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, vr), wv));
    }
#endif
    for (; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
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

static inline float dot_f32(const float *a, const float *b, int n) {
    int i = 0;
    float sum = 0.f;
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.f);
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    sum = vaddvq_f32(acc);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
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
    jval *hd = json_get(r, "head_dim");
    c->head_dim = hd ? (int)hd->num : (c->n_heads ? (c->hidden / c->n_heads) : 0);
    if (c->head_dim <= 0 && c->n_heads > 0)
        c->head_dim = c->hidden / c->n_heads;
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

static void pack_int4(const float *w, uint8_t *q4, float *scale, int O, int I) {
    const int qmax = 7;
    int rb = (I + 1) / 2;
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
        uint8_t *qr = q4 + (int64_t)o * rb;
        for (int i = 0; i < I; i += 2) {
            int v0 = (int)lrintf(wr[i] / s);
            if (v0 > qmax) v0 = qmax;
            if (v0 < -8) v0 = -8;
            int v1 = 0;
            if (i + 1 < I) {
                v1 = (int)lrintf(wr[i + 1] / s);
                if (v1 > qmax) v1 = qmax;
                if (v1 < -8) v1 = -8;
            }
            qr[i >> 1] = (uint8_t)((v0 + 8) | ((v1 + 8) << 4));
        }
    }
}

/* int4(packed)·int8 SDOT — same family as MoE path (M4: 4-acc / x86: AVX2). */
static inline int32_t dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
#if defined(__AVX2__)
    (void)i;
    return wh_dot_i4i8_avx(w4, x, I);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4q = vdupq_n_u8(0x0F);
    const int8x16_t b8q = vdupq_n_s8(8);
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    for (; i + 64 <= I; i += 64) {
        uint8x16_t byA = vld1q_u8(w4 + (i >> 1)), byB = vld1q_u8(w4 + (i >> 1) + 16);
        uint8x16x2_t zA = vzipq_u8(vandq_u8(byA, m4q), vshrq_n_u8(byA, 4));
        uint8x16x2_t zB = vzipq_u8(vandq_u8(byB, m4q), vshrq_n_u8(byB, 4));
        a0 = vdotq_s32(a0, vsubq_s8(vreinterpretq_s8_u8(zA.val[0]), b8q), vld1q_s8(x + i));
        a1 = vdotq_s32(a1, vsubq_s8(vreinterpretq_s8_u8(zA.val[1]), b8q), vld1q_s8(x + i + 16));
        a2 = vdotq_s32(a2, vsubq_s8(vreinterpretq_s8_u8(zB.val[0]), b8q), vld1q_s8(x + i + 32));
        a3 = vdotq_s32(a3, vsubq_s8(vreinterpretq_s8_u8(zB.val[1]), b8q), vld1q_s8(x + i + 48));
    }
    int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    for (; i + 32 <= I; i += 32) {
        uint8x16_t by = vld1q_u8(w4 + (i >> 1));
        uint8x16x2_t z = vzipq_u8(vandq_u8(by, m4q), vshrq_n_u8(by, 4));
        acc = vdotq_s32(acc, vsubq_s8(vreinterpretq_s8_u8(z.val[0]), b8q), vld1q_s8(x + i));
        acc = vdotq_s32(acc, vsubq_s8(vreinterpretq_s8_u8(z.val[1]), b8q), vld1q_s8(x + i + 16));
    }
    sum = vaddvq_s32(acc);
#endif
    for (; i + 1 < I; i += 2) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i] + ((int)(b >> 4) - 8) * x[i + 1];
    }
    if (i < I) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i];
    }
    return sum;
}

static void matmul_i4(float *y, const float *x, const uint8_t *q4, const float *scale, int I, int O) {
    int rb = (I + 1) / 2;
    ensure_xq(g_dens, I);
    float sx = qrow_i8(x, g_dens->xq, I);
    const int8_t *xq = g_dens->xq;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++)
        y[o] = (float)dot_i4i8(q4 + (int64_t)o * rb, xq, I) * scale[o] * sx;
}

static void matmul_i4_pair(float *yg, float *yu, const float *x,
                           const uint8_t *qg, const float *sg,
                           const uint8_t *qu, const float *su, int I, int O) {
    int rb = (I + 1) / 2;
    ensure_xq(g_dens, I);
    float sx = qrow_i8(x, g_dens->xq, I);
    const int8_t *xq = g_dens->xq;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        yg[o] = (float)dot_i4i8(qg + (int64_t)o * rb, xq, I) * sg[o] * sx;
        yu[o] = (float)dot_i4i8(qu + (int64_t)o * rb, xq, I) * su[o] * sx;
    }
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

static void load_q4weight(DModel *m, const char *name, uint8_t **q4, float **scale, int O, int I) {
    float *tmp = load_f32(m, name);
    if (!tmp) { fprintf(stderr, "dense: missing %s\n", name); exit(1); }
    int rb = (I + 1) / 2;
    *q4 = (uint8_t *)malloc((size_t)O * (size_t)rb);
    *scale = falloc(O);
    if (!*q4) { fprintf(stderr, "OOM quant4 %s\n", name); exit(1); }
    pack_int4(tmp, *q4, *scale, O, I);
    free(tmp);
}

static void dens_model_init(DModel *m, const char *snap) {
    memset(m, 0, sizeof(*m));
    dens_load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    DCfg *c = &m->c;
    m->prof = getenv("DENSE_PROF") ? 1 : 0;
    double t0 = now_s();

    float *emb = load_f32(m, "model.embed_tokens.weight");
    if (!emb) { fprintf(stderr, "dense: missing embed_tokens\n"); exit(1); }
    float *lm = load_f32(m, "lm_head.weight");
    if (!lm) {
        if (!c->tie_emb) { fprintf(stderr, "dense: missing lm_head\n"); exit(1); }
        lm = emb;
    }
    m->embed_q = (int8_t *)malloc((size_t)c->vocab * (size_t)c->hidden);
    m->embed_s = falloc(c->vocab);
    if (!m->embed_q) { fprintf(stderr, "OOM embed_q\n"); exit(1); }
    quantize_rows(emb, m->embed_q, m->embed_s, c->vocab, c->hidden);
    m->lm4 = NULL;
    m->lm4_s = NULL;
    if (lm == emb) {
        m->lm_q = NULL;
        m->lm_s = NULL;
        free(emb);
    } else {
        m->lm_q = (int8_t *)malloc((size_t)c->vocab * (size_t)c->hidden);
        m->lm_s = falloc(c->vocab);
        if (!m->lm_q) { fprintf(stderr, "OOM lm_q\n"); exit(1); }
        quantize_rows(lm, m->lm_q, m->lm_s, c->vocab, c->hidden);
        free(emb);
        free(lm);
    }

    m->final_norm = load_f32(m, "model.norm.weight");
    if (!m->final_norm) { fprintf(stderr, "dense: missing model.norm\n"); exit(1); }

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
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_norm.weight", i);
        l->q_norm = load_f32(m, nm); /* Qwen3; NULL on Qwen2/Llama */
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_norm.weight", i);
        l->k_norm = load_f32(m, nm);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", i);
        load_q4weight(m, nm, &l->o4, &l->os, D, H * hd);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", i);
        load_q4weight(m, nm, &l->gate4, &l->gates, I, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", i);
        load_q4weight(m, nm, &l->up4, &l->ups, I, D);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", i);
        load_q4weight(m, nm, &l->down4, &l->downs, D, I);
    }
    m->load_s = now_s() - t0;
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
    m->ws_sc = NULL;
    m->xq = NULL;
    m->xq_cap = 0;
    fprintf(stderr, "[dense] loaded %s in %.1fs | RSS %.2f GB | layers=%d hidden=%d | int8 qkv + int4 o/mlp\n",
            snap, m->load_s, rss_gb(), c->n_layers, c->hidden);
}

static void dens_embed(DModel *m, int token, float *out) {
    int D = m->c.hidden;
    const int8_t *qr = m->embed_q + (int64_t)token * D;
    float s = m->embed_s[token];
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vs = vdupq_n_f32(s);
    for (; i + 8 <= D; i += 8) {
        int16x8_t w16 = vmovl_s8(vld1_s8(qr + i));
        float32x4_t lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))), vs);
        float32x4_t hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))), vs);
        vst1q_f32(out + i, lo);
        vst1q_f32(out + i + 4, hi);
    }
#endif
    for (; i < D; i++) out[i] = (float)qr[i] * s;
}

static void dens_attention(DModel *m, DLayer *l, int layer, float *x, int pos, float *out) {
    DCfg *c = &m->c;
    int D = c->hidden, H = c->n_heads, KV = c->n_kv_heads, hd = c->head_dim;
    int gqa = H / KV;
    float *q = m->ws_q, *k = m->ws_k, *v = m->ws_v;
    matmul_q_ex(q, x, l->q, l->qs, D, H * hd, 0); /* exact: attn projections are IDOT-sensitive */
    matmul_q_ex(k, x, l->k, l->ks, D, KV * hd, 0);
    matmul_q_ex(v, x, l->v, l->vs, D, KV * hd, 0);
    if (l->qb) for (int i = 0; i < H * hd; i++) q[i] += l->qb[i];
    if (l->kb) for (int i = 0; i < KV * hd; i++) k[i] += l->kb[i];
    if (l->vb) for (int i = 0; i < KV * hd; i++) v[i] += l->vb[i];
    /* Qwen3: RMSNorm per head on q/k before RoPE (weight length = head_dim). */
    if (l->q_norm) {
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + hh * hd;
            rmsnorm_row(qh, qh, l->q_norm, hd, c->eps);
        }
    }
    if (l->k_norm) {
        for (int hh = 0; hh < KV; hh++) {
            float *kh = k + hh * hd;
            rmsnorm_row(kh, kh, l->k_norm, hd, c->eps);
        }
    }
    for (int hh = 0; hh < H; hh++) rope_head(q + hh * hd, pos, hd, m->rope_inv);
    for (int hh = 0; hh < KV; hh++) rope_head(k + hh * hd, pos, hd, m->rope_inv);
    for (int hh = 0; hh < KV; hh++) {
        memcpy(m->K[layer] + ((int64_t)hh * m->max_t + pos) * hd, k + hh * hd, (size_t)hd * sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh * m->max_t + pos) * hd, v + hh * hd, (size_t)hd * sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = m->ws_ctx;
    /* Parallel heads help once context is long enough to amortize OpenMP spawn. */
    #pragma omp parallel for schedule(static) if(H >= 4 && pos >= 16)
    for (int hh = 0; hh < H; hh++) {
        int kvh = hh / gqa;
        const float *qv = q + hh * hd;
        float *sc = m->ws_sc + (int64_t)hh * m->max_t;
        for (int t = 0; t <= pos; t++) {
            const float *kv = m->K[layer] + ((int64_t)kvh * m->max_t + t) * hd;
            sc[t] = dot_f32(qv, kv, hd) * scale;
        }
        softmax_row(sc, pos + 1);
        float *cx = ctx + hh * hd;
        memset(cx, 0, (size_t)hd * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *vr = m->V[layer] + ((int64_t)kvh * m->max_t + t) * hd;
            float a = sc[t];
            int d = 0;
#if defined(__ARM_NEON)
            float32x4_t va = vdupq_n_f32(a);
            for (; d + 4 <= hd; d += 4) {
                float32x4_t c0 = vld1q_f32(cx + d);
                c0 = vfmaq_f32(c0, va, vld1q_f32(vr + d));
                vst1q_f32(cx + d, c0);
            }
#endif
            for (; d < hd; d++) cx[d] += a * vr[d];
        }
    }
    matmul_i4(out, ctx, l->o4, l->os, H * hd, D); /* int4 o_proj + IDOT */
}

static void dens_mlp(DModel *m, DLayer *l, const float *x, float *out) {
    int D = m->c.hidden, I = m->c.inter;
    float *g = m->ws_g, *u = m->ws_u;
    matmul_i4_pair(g, u, x, l->gate4, l->gates, l->up4, l->ups, D, I);
    #pragma omp parallel for schedule(static) if(I >= 1024)
    for (int i = 0; i < I; i++) {
        float gv = g[i];
        /* silu(g)*u */
        g[i] = (gv / (1.f + expf(-gv))) * u[i];
    }
    matmul_i4(out, g, l->down4, l->downs, I, D);
}

static void add_inplace(float *x, const float *dx, int D) {
    int d = 0;
#if defined(__ARM_NEON)
    for (; d + 8 <= D; d += 8) {
        float32x4_t a0 = vld1q_f32(x + d), a1 = vld1q_f32(x + d + 4);
        vst1q_f32(x + d, vaddq_f32(a0, vld1q_f32(dx + d)));
        vst1q_f32(x + d + 4, vaddq_f32(a1, vld1q_f32(dx + d + 4)));
    }
#endif
    for (; d < D; d++) x[d] += dx[d];
}

static float *dens_step(DModel *m, int token, int pos) {
    DCfg *c = &m->c;
    int D = c->hidden;
    float *x = m->ws_x, *nrm = m->ws_nrm, *tmp = m->ws_tmp;
    dens_embed(m, token, x);
    for (int i = 0; i < c->n_layers; i++) {
        DLayer *l = &m->L[i];
        double t0 = m->prof ? now_s() : 0;
        rmsnorm_row(nrm, x, l->in_ln, D, c->eps);
        dens_attention(m, l, i, nrm, pos, tmp);
        add_inplace(x, tmp, D);
        if (m->prof) m->t_attn += now_s() - t0;
        t0 = m->prof ? now_s() : 0;
        rmsnorm_row(nrm, x, l->post_ln, D, c->eps);
        dens_mlp(m, l, nrm, tmp);
        add_inplace(x, tmp, D);
        if (m->prof) m->t_mlp += now_s() - t0;
    }
    m->kv_len = pos + 1;
    rmsnorm_row(nrm, x, m->final_norm, D, c->eps);
    double t0 = m->prof ? now_s() : 0;
    {
        const int8_t *lq = m->lm_q ? m->lm_q : m->embed_q;
        const float *ls = m->lm_s ? m->lm_s : m->embed_s;
        matmul_q(m->ws_logit, nrm, lq, ls, D, c->vocab);
    }
    if (m->prof) m->t_lm += now_s() - t0;
    return m->ws_logit;
}

static int argmax(const float *x, int n) {
    int b = 0;
    float v = x[0];
    for (int i = 1; i < n; i++) if (x[i] > v) { v = x[i]; b = i; }
    return b;
}

/* True when SNAP is a dense GQA+SwiGLU causal LM the dense path can run.
 * Covers Qwen2/Qwen3/Llama/Mistral and distill packs that share that layout.
 * MoE (glm_moe_dsa / routed experts) stays on engine.c — same int4+IDOT family. */
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
    if (strstr(buf, "glm_moe_dsa") || strstr(buf, "n_routed_experts") ||
        strstr(buf, "\"num_experts\"") || strstr(buf, "num_local_experts") ||
        strstr(buf, "MixtralForCausalLM") || strstr(buf, "Qwen2MoeForCausalLM") ||
        strstr(buf, "Qwen3MoeForCausalLM")) {
        free(buf);
        return 0;
    }
    /* Known dense families (incl. Qwen3 with optional q_norm/k_norm). */
    if (strstr(buf, "\"qwen2\"") || strstr(buf, "Qwen2ForCausalLM") ||
        strstr(buf, "\"qwen3\"") || strstr(buf, "Qwen3ForCausalLM") ||
        strstr(buf, "\"llama\"") || strstr(buf, "LlamaForCausalLM") ||
        strstr(buf, "\"mistral\"") || strstr(buf, "MistralForCausalLM"))
        hit = 1;
    /* Gemma / Phi use different layer contracts — keep preview until ported. */
    if (strstr(buf, "Gemma2ForCausalLM") || strstr(buf, "Gemma3ForCausalLM") ||
        strstr(buf, "\"gemma2\"") || strstr(buf, "\"gemma\"") ||
        strstr(buf, "Phi3ForCausalLM") || strstr(buf, "\"phi3\"") ||
        strstr(buf, "\"phi\""))
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
    /* Keep OpenMP workers hot across tiny matmul regions (same idea as MoE path). */
    setenv("OMP_WAIT_POLICY", "active", 0);
    setenv("OMP_PROC_BIND", "close", 0);
    setenv("OMP_MAX_ACTIVE_LEVELS", "1", 0);
    setenv("OMP_NESTED", "FALSE", 0);

    DModel m;
    dens_model_init(&m, snap);
    g_dens = &m;
    DCfg *c = &m.c;

    char tkp[2048];
    snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    Tok T;
    tok_load(&T, tkp);
    int stops[12];
    int nstop = 0;
    #define D_ADD_STOP(id) do { \
        int _id = (id); \
        if (_id >= 0) { \
            int _dup = 0; \
            for (int _i = 0; _i < nstop; _i++) if (stops[_i] == _id) { _dup = 1; break; } \
            if (!_dup && nstop < (int)(sizeof(stops)/sizeof(stops[0]))) stops[nstop++] = _id; \
        } \
    } while (0)
    D_ADD_STOP(tok_id_of(&T, "<|end|>"));
    D_ADD_STOP(tok_id_of(&T, "<|im_end|>"));
    D_ADD_STOP(tok_id_of(&T, "<|eot_id|>"));
    D_ADD_STOP(tok_id_of(&T, "<|eom_id|>"));
    D_ADD_STOP(tok_id_of(&T, "<end_of_turn>"));
    D_ADD_STOP(tok_id_of(&T, "<eos>"));
    D_ADD_STOP(tok_id_of(&T, "</s>"));
    D_ADD_STOP(tok_id_of(&T, "<|endoftext|>"));
    D_ADD_STOP(tok_id_of(&T, "<|user|>"));
    D_ADD_STOP(tok_id_of(&T, "<|system|>"));
    D_ADD_STOP(tok_id_of(&T, "<|im_start|>"));
    D_ADD_STOP(c->eos_id);
    #undef D_ADD_STOP
    /* Merge generation_config.json eos_token_id array when present. */
    {
        char gpath[2048];
        snprintf(gpath, sizeof(gpath), "%s/generation_config.json", snap);
        FILE *gf = fopen(gpath, "rb");
        if (gf) {
            fseek(gf, 0, SEEK_END);
            long gn = ftell(gf);
            fseek(gf, 0, SEEK_SET);
            if (gn > 0 && gn < 1 << 20) {
                char *gbuf = malloc((size_t)gn + 1);
                if (gbuf && fread(gbuf, 1, (size_t)gn, gf) == (size_t)gn) {
                    gbuf[gn] = 0;
                    char *arena = NULL;
                    jval *gr = json_parse(gbuf, &arena);
                    jval *eo = gr ? json_get(gr, "eos_token_id") : NULL;
                    if (eo && eo->t == J_NUM) {
                        int id = (int)eo->num;
                        if (id >= 0) {
                            int dup = 0;
                            for (int i = 0; i < nstop; i++) if (stops[i] == id) { dup = 1; break; }
                            if (!dup && nstop < 12) stops[nstop++] = id;
                        }
                    } else if (eo && eo->t == J_ARR) {
                        for (int i = 0; i < eo->len && nstop < 12; i++) {
                            if (eo->kids[i]->t != J_NUM) continue;
                            int id = (int)eo->kids[i]->num;
                            if (id < 0) continue;
                            int dup = 0;
                            for (int j = 0; j < nstop; j++) if (stops[j] == id) { dup = 1; break; }
                            if (!dup) stops[nstop++] = id;
                        }
                    }
                    free(arena);
                }
                free(gbuf);
            }
            fclose(gf);
        }
    }
    int eos = nstop > 0 ? stops[0] : -1;
    (void)eos;

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
    m.ws_sc = falloc((int64_t)c->n_heads * m.max_t); /* per-head score rows for OpenMP */
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

    if (m.prof) m.t_attn = m.t_mlp = m.t_lm = 0;
    double t0 = now_s();
    int generated = 0;
    char outbuf[4096];
    for (int s = 0; s < ngen; s++) {
        int tok = argmax(logit, c->vocab);
        generated++;
        /* Stop before emit so control tokens like <|end|> / <|im_end|> never hit stdout. */
        int hit = 0;
        for (int i = 0; i < nstop; i++) if (stops[i] == tok) { hit = 1; break; }
        if (hit) break;
        int nch = tok_decode(&T, &tok, 1, outbuf, (int)sizeof(outbuf) - 1);
        if (nch > 0) {
            outbuf[nch] = 0;
            fputs(outbuf, stdout);
            fflush(stdout);
        }
        if (s + 1 == ngen) break;
        logit = dens_step(&m, tok, np + s);
    }
    fputc('\n', stdout);
    fflush(stdout);
    double dt = now_s() - t0;
    double tps = dt > 0 ? (generated / dt) : 0;
    fprintf(stderr, "[dense] decode %.2f tok/s (%.2fs for %d toks) | RSS %.2f GB | load %.1fs\n",
            tps, dt, generated, rss_gb(), m.load_s);
    if (m.prof && generated > 0)
        fprintf(stderr, "[dense][prof] attn=%.2fs mlp=%.2fs lm=%.2fs (decode window)\n",
                m.t_attn, m.t_mlp, m.t_lm);
    free(prompt_ids);
    g_dens = NULL;
    return 0;
}
