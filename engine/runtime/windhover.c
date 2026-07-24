/* windhover.c — Windhover: unified sparse working-set runtime (dense KPK packs).
 *
 * Consumes packs built by tools/kestrel_pack.py:
 *   - weights pre-quantized on disk (int8 per-row / int4 group-64 asymmetric,
 *     fp16 group scales+zeros), mmap'd zero-copy (clean, evictable pages)
 *   - descriptor-driven forward (qwen2/qwen3/llama/mistral/gemma2/gemma3/phi3)
 *   - int8 KV cache (group-32 within head_dim, fp16 scales)
 *   - batched prefill: weights stream once per S-token chunk
 *   - CATS sparse FFN: full gate, threshold |act(g)|, skip up/down^T rows
 *   - AU tier manager: hot-prefix bundles resident/mlocked, cold tail on SSD
 *   - opt-in n-gram self-speculation (WH_SPEC=1, greedy-exact)
 *
 * Quality profile validated by tools/windhover_gates.py (G2, "wh_c"):
 * int8-row qkv+lm, int4-g64-asym o/gate/up/down^T with AWQ folded at convert.
 *
 * Env: SNAP PROMPT|COLI_PROMPT NGEN TEMP TOPK TOPP NUCLEUS SEED CTX QUIET
 *      RAM_GB WH_SPARSE(0|25|40) WH_SPEC(0|1) WH_STATS(1) MLOCK IDOT
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#if defined(_WIN32)
/* getrusage / setenv come from compat.h via st.h */
#elif defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
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
#include "json.h"
#include "model_desc.h"
#include "wmir.h"
#include "au.h"
#include "budget.h"
#include "windhover.h"

#define WH_GS 64          /* weight quant group */
#define WH_KVG 32         /* kv quant group */
#define WH_PREFILL_S 80   /* prefill chunk (was 64; matches WH_MAXS headroom) */
#define WH_MAXS 80        /* max batch rows in scratch (prefill/verify) */
#define WH_MAX_HD 512     /* max head_dim for stack q8 buffer */
#define WH_MAX_HIDDEN 8192

/* Portable IEEE fp16 storage. Apple Clang accepts __fp16 by value; Linux
 * x86 Clang/GCC reject that, so prefer _Float16 when the compiler provides it. */
#if defined(__FLT16_MAX__)
typedef _Float16 wh_f16;
#elif defined(__ARM_FP16_FORMAT_IEEE) || defined(__fp16)
typedef __fp16 wh_f16;
#else
#error "Windhover requires IEEE fp16 (_Float16 or __fp16)"
#endif


/* ------------------------------------------------------------------ utils */

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double rss_gb(void) {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
#if defined(__APPLE__)
    return r.ru_maxrss / (1024.0 * 1024.0 * 1024.0);
#else
    return r.ru_maxrss / (1024.0 * 1024.0);
#endif
}
static double footprint_gb(void) {
#if defined(__APPLE__)
    task_vm_info_data_t vm;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &cnt) == KERN_SUCCESS)
        return (double)vm.phys_footprint / 1e9;
#endif
    return rss_gb();
}
static float *falloc(int64_t n) {
    float *p = calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "[wh] OOM %lld floats\n", (long long)n); exit(1); }
    return p;
}
static void *balloc(int64_t n) {
    void *p = calloc((size_t)n, 1);
    if (!p) { fprintf(stderr, "[wh] OOM %lld bytes\n", (long long)n); exit(1); }
    return p;
}
static int wh_is_stop(const int *stops, int nstop, int tok) {
    for (int i = 0; i < nstop; i++) if (stops[i] == tok) return 1;
    return 0;
}

/* Pull every eos_token_id from config.json / generation_config.json (scalar or array). */
static void wh_add_eos_from_json_file(const char *path, int *stops, int *nstop, int cap) {
    long n = 0;
    char *buf = wh_read_file_(path, &n);
    if (!buf) return;
    char *arena = NULL;
    jval *r = json_parse(buf, &arena);
    if (!r) { free(buf); return; }
    jval *eo = json_get(r, "eos_token_id");
    if (eo) {
        if (eo->t == J_NUM) {
            int id = (int)eo->num;
            if (id >= 0 && *nstop < cap && !wh_is_stop(stops, *nstop, id))
                stops[(*nstop)++] = id;
        } else if (eo->t == J_ARR) {
            for (int i = 0; i < eo->len && *nstop < cap; i++) {
                if (eo->kids[i]->t != J_NUM) continue;
                int id = (int)eo->kids[i]->num;
                if (id >= 0 && !wh_is_stop(stops, *nstop, id))
                    stops[(*nstop)++] = id;
            }
        }
    }
    free(buf);
    free(arena);
}

static void wh_arm_chat_stops(Tok *T, WhDesc *d, const char *snap,
                              int *stops, int *nstop, int cap) {
    *nstop = 0;
    #define WH_ADD_STOP(id) do { \
        int _id = (id); \
        if (_id >= 0 && *nstop < cap && !wh_is_stop(stops, *nstop, _id)) \
            stops[(*nstop)++] = _id; \
    } while (0)
    /* Family-specific chat end markers first (critical for Phi / Llama / Gemma). */
    WH_ADD_STOP(tok_id_of(T, "<|end|>"));
    WH_ADD_STOP(tok_id_of(T, "<|im_end|>"));
    WH_ADD_STOP(tok_id_of(T, "<|eot_id|>"));
    WH_ADD_STOP(tok_id_of(T, "<|eom_id|>"));
    WH_ADD_STOP(tok_id_of(T, "<end_of_turn>"));
    WH_ADD_STOP(tok_id_of(T, "<eos>"));
    WH_ADD_STOP(tok_id_of(T, "</s>"));
    WH_ADD_STOP(tok_id_of(T, "<|endoftext|>"));
    /* Also stop if the model starts the next role (runaway after missed EOS). */
    WH_ADD_STOP(tok_id_of(T, "<|user|>"));
    WH_ADD_STOP(tok_id_of(T, "<|system|>"));
    WH_ADD_STOP(tok_id_of(T, "<|im_start|>"));
    WH_ADD_STOP(d->eos_id);
    #undef WH_ADD_STOP
    /* Merge every eos_token_id from HF configs (Phi lists [<|end|>, <|endoftext|>]). */
    char path[2048];
    snprintf(path, sizeof(path), "%s/generation_config.json", snap);
    wh_add_eos_from_json_file(path, stops, nstop, cap);
    snprintf(path, sizeof(path), "%s/config.json", snap);
    wh_add_eos_from_json_file(path, stops, nstop, cap);
}

/* ------------------------------------------------------------- weight refs */

typedef enum { WT_NONE = 0, WT_F32, WT_I8R, WT_I4G } WtFmt;

typedef struct {
    WtFmt fmt;
    int O, I, ng;
    const int8_t *q8;      /* WT_I8R */
    const float *rs;       /* WT_I8R row scales (f32) */
    const uint8_t *q4;     /* WT_I4G packed nibbles (q+8) */
    const wh_f16 *sc;      /* WT_I4G group scales */
    const wh_f16 *zp;      /* WT_I4G group zeros */
    const float *f;        /* WT_F32 */
    const wh_f16 *oc;      /* fp16 outlier columns [O][noc] (down^T) */
    const int32_t *oci;    /* outlier column indices [noc] */
    int noc;
    int64_t bytes;         /* weight+scale bytes (telemetry) */
} WT;

typedef struct {
    const float *in_ln, *post_ln;      /* norms (f32 views) */
    const float *pre_ffn_ln, *post_ffn_ln; /* gemma sandwich */
    const float *q_norm, *k_norm;
    const float *qb, *kb, *vb;         /* biases */
    WT q, k, v, o, gate, up, downT;
    int is_sw;                         /* sliding-window layer */
    /* WMIR per-layer contract */
    WmirOpKind attn_kind;
    WmirOpKind mlp_kind;
    int kv_share_from;                 /* -1 = own KV */
    int chunk_size;                    /* attn_chunked / msa block */
    int has_gqa;                       /* 1 if q/k/v/o present */
    int has_linear;                    /* 1 if linear_gdn tensors present */
    int layer_inter;                   /* MLP width (may differ per layer) */
    /* linear GDN (optional f32 / quant views) */
    WT lin_qkv, lin_out;
    const float *lin_A_log, *lin_dt_bias, *lin_norm;
    int lin_dim;                       /* in_proj out rows / 3 approx */
} WLayer;

typedef struct {
    WhDesc d;
    WmirGraph wmir;
    shards S;
    WT embed, lm;
    const float *final_norm;
    WLayer *L;
    /* kv cache int8 g32 */
    int8_t **K8, **V8;                 /* [l] -> [kvh][max_t][hd] */
    wh_f16 **KS, **VS;                 /* [l] -> [kvh][max_t][hd/WH_KVG] */
    int max_t, kv_len;
    /* rope table: [max_t][rope_half] where rope_half = rope_dim/2 */
    float *rope_cos, *rope_sin;
    int rope_dim, rope_half;
    /* scratch */
    float *x, *nrm, *tmp, *q, *k, *v, *ctx, *sc, *g, *u, *logit; /* batched: [S][*] */
    int8_t *xq; float *sx; int32_t *xqsum;   /* activation quant [S][I], [S], [S][ng] */
    float *dacc;                        /* down accum per thread [T][S][D] */
    float *lin_state;                   /* [layers][lin_dim] recurrent state */
    int nthreads;
    /* sparsity */
    float *cats_tau;                    /* [layers] chosen tau (0=off) */
    float tau_scale_cold;
    int *Jlist; float *Jmag; unsigned char *Jkeep;
    /* online tau calibration (packs without calibrated thresholds) */
    float *tau_res;                     /* [layers][WH_TAU_RES] |act| reservoir */
    int *tau_n;                         /* samples collected per layer */
    int tau_online, tau_target_pct;
    AuPool au;
    /* telemetry */
    int64_t bytes_full, bytes_read;     /* per-token model bytes: nominal vs after skips */
    int64_t ffn_rows_total, ffn_rows_kept;
    double t_attn, t_mlp, t_lm;
    double load_s;
    int prof;
} WModel;

static WModel *g_wh;

/* ----------------------------------------------------------- int8 kernels */

static inline int32_t dot_i8i8(const int8_t *w, const int8_t *x, int I) {
    int32_t sum = 0; int i = 0;
#if defined(__AVX2__)
    (void)i;
    return wh_dot_i8i8_avx(w, x, I);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    for (; i + 64 <= I; i += 64) {
        a0 = vdotq_s32(a0, vld1q_s8(w + i),      vld1q_s8(x + i));
        a1 = vdotq_s32(a1, vld1q_s8(w + i + 16), vld1q_s8(x + i + 16));
        a2 = vdotq_s32(a2, vld1q_s8(w + i + 32), vld1q_s8(x + i + 32));
        a3 = vdotq_s32(a3, vld1q_s8(w + i + 48), vld1q_s8(x + i + 48));
    }
    int32x4_t acc = vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3));
    for (; i + 16 <= I; i += 16) acc = vdotq_s32(acc, vld1q_s8(w + i), vld1q_s8(x + i));
    sum = vaddvq_s32(acc);
#endif
    for (; i < I; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}

/* int4-g64-asym row dot with precomputed per-group activation sums:
 * y = sx * ( sum_g sc[g]*idot(q_g, xq_g) + zp[g]*xqsum[g] ) */
static inline float dot_i4g(const uint8_t *w4, const wh_f16 *sc, const wh_f16 *zp,
                            const int8_t *xq, const int32_t *xqsum, float sx, int I) {
    float acc = 0.f;
#if defined(__AVX2__)
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        const int8_t *xg = xq + g * WH_GS;
        int32_t p = wh_dot_i4i8_avx(wg, xg, WH_GS);
        acc += (float)sc[g] * (float)p + (float)zp[g] * (float)xqsum[g];
    }
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4q = vdupq_n_u8(0x0F);
    const int8x16_t b8q = vdupq_n_s8(8);
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        const int8_t *xg = xq + g * WH_GS;
        uint8x16_t byA = vld1q_u8(wg), byB = vld1q_u8(wg + 16);
        uint8x16x2_t zA = vzipq_u8(vandq_u8(byA, m4q), vshrq_n_u8(byA, 4));
        uint8x16x2_t zB = vzipq_u8(vandq_u8(byB, m4q), vshrq_n_u8(byB, 4));
        int32x4_t p = vdupq_n_s32(0);
        p = vdotq_s32(p, vsubq_s8(vreinterpretq_s8_u8(zA.val[0]), b8q), vld1q_s8(xg));
        p = vdotq_s32(p, vsubq_s8(vreinterpretq_s8_u8(zA.val[1]), b8q), vld1q_s8(xg + 16));
        p = vdotq_s32(p, vsubq_s8(vreinterpretq_s8_u8(zB.val[0]), b8q), vld1q_s8(xg + 32));
        p = vdotq_s32(p, vsubq_s8(vreinterpretq_s8_u8(zB.val[1]), b8q), vld1q_s8(xg + 48));
        acc += (float)sc[g] * (float)vaddvq_s32(p) + (float)zp[g] * (float)xqsum[g];
    }
#else
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        int32_t p = 0;
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        const int8_t *xg = xq + g * WH_GS;
        for (int i = 0; i < WH_GS; i += 2) {
            uint8_t b = wg[i >> 1];
            p += ((int)(b & 0xF) - 8) * xg[i] + ((int)(b >> 4) - 8) * xg[i + 1];
        }
        acc += (float)sc[g] * (float)p + (float)zp[g] * (float)xqsum[g];
    }
#endif
    return acc * sx;
}

/* fused axpy: acc[d] += hj * (sc_g*q[d] + zp_g), single pass, no scratch */
static inline void axpy_i4g_row(float *acc, const uint8_t *w4, const wh_f16 *sc,
                                const wh_f16 *zp, float hj, int I) {
#if defined(__AVX2__)
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        float vs = hj * (float)sc[g], vz = hj * (float)zp[g];
        float *o = acc + g * WH_GS;
        wh_axpy_i4g32_avx(o, wg, vs, vz);
        wh_axpy_i4g32_avx(o + 32, wg + 16, vs, vz);
    }
#elif defined(__ARM_NEON)
    const uint8x16_t m4q = vdupq_n_u8(0x0F);
    const int8x16_t b8q = vdupq_n_s8(8);
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        float32x4_t vs = vdupq_n_f32(hj * (float)sc[g]);
        float32x4_t vz = vdupq_n_f32(hj * (float)zp[g]);
        float *o = acc + g * WH_GS;
        for (int i = 0; i < WH_GS; i += 32) {
            uint8x16_t by = vld1q_u8(wg + (i >> 1));
            uint8x16x2_t zz = vzipq_u8(vandq_u8(by, m4q), vshrq_n_u8(by, 4));
            int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(zz.val[0]), b8q);
            int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(zz.val[1]), b8q);
            int16x8_t w0 = vmovl_s8(vget_low_s8(q0)), w1 = vmovl_s8(vget_high_s8(q0));
            int16x8_t w2 = vmovl_s8(vget_low_s8(q1)), w3 = vmovl_s8(vget_high_s8(q1));
            float *p = o + i;
            float32x4_t a0 = vld1q_f32(p),      a1 = vld1q_f32(p + 4);
            float32x4_t a2 = vld1q_f32(p + 8),  a3 = vld1q_f32(p + 12);
            float32x4_t a4 = vld1q_f32(p + 16), a5 = vld1q_f32(p + 20);
            float32x4_t a6 = vld1q_f32(p + 24), a7 = vld1q_f32(p + 28);
            a0 = vfmaq_f32(vaddq_f32(a0, vz), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))), vs);
            a1 = vfmaq_f32(vaddq_f32(a1, vz), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), vs);
            a2 = vfmaq_f32(vaddq_f32(a2, vz), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))), vs);
            a3 = vfmaq_f32(vaddq_f32(a3, vz), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), vs);
            a4 = vfmaq_f32(vaddq_f32(a4, vz), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2))), vs);
            a5 = vfmaq_f32(vaddq_f32(a5, vz), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2))), vs);
            a6 = vfmaq_f32(vaddq_f32(a6, vz), vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3))), vs);
            a7 = vfmaq_f32(vaddq_f32(a7, vz), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3))), vs);
            vst1q_f32(p, a0);      vst1q_f32(p + 4, a1);
            vst1q_f32(p + 8, a2);  vst1q_f32(p + 12, a3);
            vst1q_f32(p + 16, a4); vst1q_f32(p + 20, a5);
            vst1q_f32(p + 24, a6); vst1q_f32(p + 28, a7);
        }
    }
#else
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        float s = hj * (float)sc[g], z = hj * (float)zp[g];
        for (int i = 0; i < WH_GS; i += 2) {
            uint8_t b = wg[i >> 1];
            acc[g * WH_GS + i]     += s * (float)((int)(b & 0xF) - 8) + z;
            acc[g * WH_GS + i + 1] += s * (float)((int)(b >> 4) - 8) + z;
        }
    }
#endif
}

/* int8-row axpy: acc += hj * rs * q8[0..I) */
static inline void axpy_i8_row(float *acc, const int8_t *q8, float rs, float hj, int I) {
    float s = hj * rs;
#if defined(__AVX2__)
    wh_axpy_i8_avx(acc, q8, s, I);
#else
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vs = vdupq_n_f32(s);
    for (; i + 16 <= I; i += 16) {
        int8x16_t q = vld1q_s8(q8 + i);
        int16x8_t w0 = vmovl_s8(vget_low_s8(q)), w1 = vmovl_s8(vget_high_s8(q));
        float32x4_t a0 = vld1q_f32(acc + i), a1 = vld1q_f32(acc + i + 4);
        float32x4_t a2 = vld1q_f32(acc + i + 8), a3 = vld1q_f32(acc + i + 12);
        a0 = vfmaq_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))), vs);
        a1 = vfmaq_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), vs);
        a2 = vfmaq_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))), vs);
        a3 = vfmaq_f32(a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), vs);
        vst1q_f32(acc + i, a0); vst1q_f32(acc + i + 4, a1);
        vst1q_f32(acc + i + 8, a2); vst1q_f32(acc + i + 12, a3);
    }
#endif
    for (; i < I; i++) acc[i] += s * (float)q8[i];
#endif
}

static inline void deq_i8_row(const int8_t *q8, float rs, float *out, int I) {
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vs = vdupq_n_f32(rs);
    for (; i + 16 <= I; i += 16) {
        int8x16_t q = vld1q_s8(q8 + i);
        int16x8_t w0 = vmovl_s8(vget_low_s8(q)), w1 = vmovl_s8(vget_high_s8(q));
        vst1q_f32(out + i, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))), vs));
        vst1q_f32(out + i + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), vs));
        vst1q_f32(out + i + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))), vs));
        vst1q_f32(out + i + 12, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), vs));
    }
#endif
    for (; i < I; i++) out[i] = rs * (float)q8[i];
}

/* dequant one int4-g64 row into f32 (for down^T axpy) */
static inline void deq_i4g_row(const uint8_t *w4, const wh_f16 *sc, const wh_f16 *zp,
                               float *out, int I) {
#if defined(__ARM_NEON)
    const uint8x16_t m4q = vdupq_n_u8(0x0F);
    const int8x16_t b8q = vdupq_n_s8(8);
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        float s = (float)sc[g], z = (float)zp[g];
        float32x4_t vs = vdupq_n_f32(s), vz = vdupq_n_f32(z);
        for (int i = 0; i < WH_GS; i += 32) {
            uint8x16_t by = vld1q_u8(wg + (i >> 1));
            uint8x16x2_t zz = vzipq_u8(vandq_u8(by, m4q), vshrq_n_u8(by, 4));
            int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(zz.val[0]), b8q);
            int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(zz.val[1]), b8q);
            int16x8_t w0 = vmovl_s8(vget_low_s8(q0)), w1 = vmovl_s8(vget_high_s8(q0));
            int16x8_t w2 = vmovl_s8(vget_low_s8(q1)), w3 = vmovl_s8(vget_high_s8(q1));
            float *o = out + g * WH_GS + i;
            vst1q_f32(o,      vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))), vs));
            vst1q_f32(o + 4,  vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0))), vs));
            vst1q_f32(o + 8,  vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))), vs));
            vst1q_f32(o + 12, vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1))), vs));
            vst1q_f32(o + 16, vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2))), vs));
            vst1q_f32(o + 20, vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2))), vs));
            vst1q_f32(o + 24, vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3))), vs));
            vst1q_f32(o + 28, vfmaq_f32(vz, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3))), vs));
        }
    }
#else
    int ng = I / WH_GS;
    for (int g = 0; g < ng; g++) {
        const uint8_t *wg = w4 + ((int64_t)g * WH_GS >> 1);
        float s = (float)sc[g], z = (float)zp[g];
        for (int i = 0; i < WH_GS; i += 2) {
            uint8_t b = wg[i >> 1];
            out[g * WH_GS + i]     = s * (float)((int)(b & 0xF) - 8) + z;
            out[g * WH_GS + i + 1] = s * (float)((int)(b >> 4) - 8) + z;
        }
    }
#endif
}

/* quantize one activation row + per-group sums */
static float qrow(const float *x, int8_t *q, int32_t *gsum, int I) {
    float amax = 0.f;
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t am = vdupq_n_f32(0.f);
    for (; i + 4 <= I; i += 4) am = vmaxq_f32(am, vabsq_f32(vld1q_f32(x + i)));
    amax = vmaxvq_f32(am);
#endif
    for (; i < I; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    float s = amax / 127.f;
    if (s < 1e-12f) s = 1e-12f;
    float inv = 1.f / s;
    for (i = 0; i < I; i++) q[i] = (int8_t)lrintf(x[i] * inv);
    if (gsum) {
        int ng = I / WH_GS;
        for (int g = 0; g < ng; g++) {
            int32_t t = 0;
            const int8_t *qg = q + g * WH_GS;
            for (int j = 0; j < WH_GS; j++) t += qg[j];
            gsum[g] = t;
        }
        for (int g = ng * WH_GS; g < I; g++) { /* ragged tail unused (I%64==0 enforced) */ }
    }
    return s;
}

#if defined(WH_SME) && defined(__ARM_FEATURE_SME2)
#include <arm_sme.h>
/* SME2 SMOPA GEMM — same panel layout as tools/wh_kernel_bench.c G5.
 * Wp[opanel]: [k/4][row16][4]; Xp: [k/4][tile][col16][4]. Max S=64 (4 ZA tiles). */
__arm_locally_streaming __arm_new("za")
static void sme_gemm_panel(const int8_t *Wp, const int8_t *Xp, int32_t *Ct,
                           int I, int ncols) {
    svbool_t pg = svptrue_b8();
    int ntile = ncols / 16;
    svzero_za();
    for (int k4 = 0; k4 < I / 4; k4++) {
        svint8_t wv = svld1_s8(pg, Wp + (int64_t)k4 * 64);
        for (int t = 0; t < ntile; t++) {
            svint8_t xv = svld1_s8(pg, Xp + ((int64_t)k4 * ntile + t) * 64);
            switch (t) {
            case 0: svmopa_za32_s8_m(0, svptrue_b8(), svptrue_b8(), wv, xv); break;
            case 1: svmopa_za32_s8_m(1, svptrue_b8(), svptrue_b8(), wv, xv); break;
            case 2: svmopa_za32_s8_m(2, svptrue_b8(), svptrue_b8(), wv, xv); break;
            case 3: svmopa_za32_s8_m(3, svptrue_b8(), svptrue_b8(), wv, xv); break;
            }
        }
    }
    svbool_t pw = svptrue_b32();
    for (int row = 0; row < 16; row++) {
        for (int t = 0; t < ntile; t++) {
            svint32_t v;
            switch (t) {
            case 0: v = svread_hor_za32_s32_m(svdup_s32(0), pw, 0, row); break;
            case 1: v = svread_hor_za32_s32_m(svdup_s32(0), pw, 1, row); break;
            case 2: v = svread_hor_za32_s32_m(svdup_s32(0), pw, 2, row); break;
            default: v = svread_hor_za32_s32_m(svdup_s32(0), pw, 3, row); break;
            }
            svst1_s32(pw, Ct + ((int64_t)row * ncols) + t * 16, v);
        }
    }
}

/* Prefill-only: int8 WT_I8R with S∈{16,32,48,64}. Returns 1 if handled. */
static int mm_wt_sme_i8(const WT *w, const int8_t *xq, const float *sx,
                        float *y, int S, int ystride) {
    int O = w->O, I = w->I;
    if (w->fmt != WT_I8R || S < 16 || S > 64 || (S % 16) || (O % 16) || (I % 4))
        return 0;
    int ntile = S / 16;
    int8_t *Wp = aligned_alloc(64, (size_t)O * (size_t)I);
    int8_t *Xp = aligned_alloc(64, (size_t)(I / 4) * (size_t)ntile * 64);
    int32_t *Ct = aligned_alloc(64, (size_t)16 * (size_t)S * sizeof(int32_t));
    if (!Wp || !Xp || !Ct) { free(Wp); free(Xp); free(Ct); return 0; }
    for (int op = 0; op < O; op += 16)
        for (int k4 = 0; k4 < I / 4; k4++)
            for (int row = 0; row < 16; row++)
                for (int b = 0; b < 4; b++)
                    Wp[(int64_t)op * I + (int64_t)k4 * 64 + row * 4 + b] =
                        w->q8[(int64_t)(op + row) * I + k4 * 4 + b];
    for (int k4 = 0; k4 < I / 4; k4++)
        for (int t = 0; t < ntile; t++)
            for (int col = 0; col < 16; col++)
                for (int b = 0; b < 4; b++)
                    Xp[((int64_t)k4 * ntile + t) * 64 + col * 4 + b] =
                        xq[(int64_t)(t * 16 + col) * I + k4 * 4 + b];
    for (int op = 0; op < O; op += 16) {
        sme_gemm_panel(Wp + (int64_t)op * I, Xp, Ct, I, S);
        for (int row = 0; row < 16; row++) {
            float rs = w->rs[op + row];
            for (int s = 0; s < S; s++)
                y[(int64_t)s * ystride + op + row] =
                    (float)Ct[row * S + s] * rs * sx[s];
        }
    }
    free(Wp); free(Xp); free(Ct);
    return 1;
}
#endif

/* Batched GEMV: y[s][o]. Weight rows stream once; s inner. Inside omp region. */
static void mm_wt(const WT *w, const int8_t *xq, const float *sx,
                  const int32_t *xqsum, float *y, int S, int ystride) {
    int O = w->O, I = w->I, ng = w->ng;
    if (w->fmt == WT_I8R) {
        #pragma omp for schedule(static) nowait
        for (int o = 0; o < O; o++) {
            const int8_t *wr = w->q8 + (int64_t)o * I;
            float rs = w->rs[o];
            for (int s = 0; s < S; s++)
                y[(int64_t)s * ystride + o] = (float)dot_i8i8(wr, xq + (int64_t)s * I, I) * rs * sx[s];
        }
    } else if (w->fmt == WT_I4G) {
        #pragma omp for schedule(static) nowait
        for (int o = 0; o < O; o++) {
            const uint8_t *wr = w->q4 + (int64_t)o * (I >> 1);
            const wh_f16 *sc = w->sc + (int64_t)o * ng;
            const wh_f16 *zp = w->zp + (int64_t)o * ng;
            for (int s = 0; s < S; s++)
                y[(int64_t)s * ystride + o] =
                    dot_i4g(wr, sc, zp, xq + (int64_t)s * I, xqsum + (int64_t)s * ng, sx[s], I);
        }
    } else {
        #pragma omp for schedule(static) nowait
        for (int o = 0; o < O; o++) {
            const float *wr = w->f + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                /* f32 fallback path is only for tiny tensors */
                float acc = 0.f;
                const int8_t *xr = xq + (int64_t)s * I;
                for (int i = 0; i < I; i++) acc += wr[i] * (float)xr[i];
                y[(int64_t)s * ystride + o] = acc * sx[s];
            }
        }
    }
}

/* Outside OpenMP: optional SME2 int8 prefill, else OMP GEMV.
 * SME packs weights on the fly (microbench G5); that tax dominates until
 * panels are cached at load — opt in with WH_SME_RUNTIME=1 after SME=1 build. */
static void mm_wt_run(const WT *w, const int8_t *xq, const float *sx,
                      const int32_t *xqsum, float *y, int S, int ystride) {
#if defined(WH_SME) && defined(__ARM_FEATURE_SME2)
    static int sme_rt = -1;
    if (sme_rt < 0) sme_rt = getenv("WH_SME_RUNTIME") ? 1 : 0;
    if (sme_rt && S >= 16 && mm_wt_sme_i8(w, xq, sx, y, S, ystride))
        return;
#endif
#ifdef _OPENMP
    if (omp_in_parallel()) {
        mm_wt(w, xq, sx, xqsum, y, S, ystride);
        return;
    }
#endif
    #pragma omp parallel
    { mm_wt(w, xq, sx, xqsum, y, S, ystride); }
}

/* ------------------------------------------------------------- norms/math */

static void rmsnorm_row(float *out, const float *x, const float *w, int D,
                        float eps, int gemma) {
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
    if (!gemma) {
        for (i = 0; i < D; i++) out[i] = x[i] * r * w[i];
    } else {
        for (i = 0; i < D; i++) out[i] = x[i] * r * (1.f + w[i]);
    }
}

static void softmax_row(float *x, int n) {
    float m = -1e30f;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv = 1.f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static inline float act_fn(float g, int gelu) {
    if (!gelu) return g / (1.f + expf(-g));                    /* silu */
    /* gelu tanh approx */
    float c = 0.7978845608f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (1.f + tanhf(c));
}

#if defined(__ARM_NEON)
/* Cephes-style vectorized exp (rel err ~2e-7), the standard ggml/llama.cpp
 * polynomial. Scalar expf dominated decode profile before this. */
static inline float32x4_t vexpq_f32(float32x4_t x) {
    const float32x4_t LOG2E = vdupq_n_f32(1.442695040f);
    const float32x4_t C0 = vdupq_n_f32(0.693359375f);
    const float32x4_t C1 = vdupq_n_f32(-2.12194440e-4f);
    x = vminq_f32(vdupq_n_f32(88.3762626647949f), vmaxq_f32(vdupq_n_f32(-88.3762626647949f), x));
    float32x4_t fx = vrndnq_f32(vmulq_f32(x, LOG2E));
    x = vfmsq_f32(x, fx, C0);
    x = vfmsq_f32(x, fx, C1);
    float32x4_t z = vmulq_f32(x, x);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vfmaq_f32(vfmaq_f32(vdupq_n_f32(1.f), x, vdupq_n_f32(1.f)), y, z);
    int32x4_t n = vcvtq_s32_f32(fx);
    n = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(n));
}
#endif

/* g[i] = act(g[i]) for i in [0,n) — bulk, vectorized. */
static void act_bulk(float *g, int n, int gelu) {
    int i = 0;
#if defined(__ARM_NEON)
    if (!gelu) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(g + i);
            float32x4_t e = vexpq_f32(vnegq_f32(v));
            float32x4_t r = vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.f), e));
            vst1q_f32(g + i, r);
        }
    } else {
        /* gelu_tanh(x) = x * sigmoid(1.702x) is a poorer fit; use exact-ish
         * tanh form via exp: tanh(c) = 1 - 2/(e^{2c}+1) */
        const float32x4_t k0 = vdupq_n_f32(0.7978845608f);
        const float32x4_t k1 = vdupq_n_f32(0.044715f);
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(g + i);
            float32x4_t c = vmulq_f32(k0, vfmaq_f32(v, vmulq_f32(vmulq_f32(v, v), v), k1));
            float32x4_t e = vexpq_f32(vmulq_f32(c, vdupq_n_f32(2.f)));
            float32x4_t th = vsubq_f32(vdupq_n_f32(1.f),
                                       vdivq_f32(vdupq_n_f32(2.f),
                                                 vaddq_f32(e, vdupq_n_f32(1.f))));
            vst1q_f32(g + i, vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), v),
                                       vaddq_f32(vdupq_n_f32(1.f), th)));
        }
    }
#endif
    for (; i < n; i++) g[i] = act_fn(g[i], gelu);
}

static void add_inplace(float *x, const float *dx, int D) {
    int d = 0;
#if defined(__ARM_NEON)
    for (; d + 4 <= D; d += 4)
        vst1q_f32(x + d, vaddq_f32(vld1q_f32(x + d), vld1q_f32(dx + d)));
#endif
    for (; d < D; d++) x[d] += dx[d];
}

/* --------------------------------------------------------------- loading */

static void wt_from_view(WModel *m, const char *name, WT *w, int O, int I, int want_i8) {
    st_view v, vs, vz;
    char nm[320];
    memset(w, 0, sizeof(*w));
    if (!st_view_get(&m->S, name, &v)) { w->fmt = WT_NONE; return; }
    /* Infer O when caller passes 0 (linear_attn / variable shapes). */
    if (O <= 0 && I > 0) {
        snprintf(nm, sizeof(nm), "%s.qs", name);
        int has_s = st_view_get(&m->S, nm, &vs);
        snprintf(nm, sizeof(nm), "%s.qz", name);
        int has_z = st_view_get(&m->S, nm, &vz);
        if (has_s && has_z && I >= 2)
            O = (int)(v.nbytes / (I / 2));
        else if (has_s)
            O = (int)(v.nbytes / I);
        else if (v.dtype == 2)
            O = (int)(v.numel / I);
        if (O <= 0) { w->fmt = WT_NONE; return; }
    }
    w->O = O; w->I = I; w->ng = I / WH_GS;
    snprintf(nm, sizeof(nm), "%s.qs", name);
    int has_s = st_view_get(&m->S, nm, &vs);
    snprintf(nm, sizeof(nm), "%s.qz", name);
    int has_z = st_view_get(&m->S, nm, &vz);
    if (has_s && has_z && v.nbytes == (int64_t)O * (I / 2)) {
        w->fmt = WT_I4G;
        w->q4 = (const uint8_t *)v.p;
        w->sc = (const wh_f16 *)vs.p;
        w->zp = (const wh_f16 *)vz.p;
        w->bytes = v.nbytes + vs.nbytes + vz.nbytes;
        st_view vo, voi;
        snprintf(nm, sizeof(nm), "%s.oc", name);
        if (st_view_get(&m->S, nm, &vo)) {
            snprintf(nm, sizeof(nm), "%s.oci", name);
            if (st_view_get(&m->S, nm, &voi)) {
                w->oc = (const wh_f16 *)vo.p;
                w->oci = (const int32_t *)voi.p;
                w->noc = (int)voi.numel;
                w->bytes += vo.nbytes + voi.nbytes;
            }
        }
    } else if (has_s && v.nbytes == (int64_t)O * I) {
        w->fmt = WT_I8R;
        w->q8 = (const int8_t *)v.p;
        w->rs = (const float *)vs.p;
        w->bytes = v.nbytes + vs.nbytes;
    } else if (v.dtype == 2 && v.numel == (int64_t)O * I) {
        w->fmt = WT_F32;
        w->f = (const float *)v.p;
        w->bytes = v.nbytes;
    } else if (v.dtype == 2 || v.dtype == 0 || v.dtype == 1) {
        /* Raw f32/bf16 stored without .qs (linear_attn side tensors). */
        w->fmt = WT_F32;
        w->f = (const float *)v.p;
        w->O = O > 0 ? O : (int)v.numel;
        w->I = I > 0 ? I : 1;
        w->bytes = v.nbytes;
    } else {
        fprintf(stderr, "[wh] tensor %s: unexpected format (nbytes=%lld dtype=%d)\n",
                name, (long long)v.nbytes, v.dtype);
        exit(1);
    }
    (void)want_i8;
}

static const float *f32_view(WModel *m, const char *name, int64_t expect) {
    st_view v;
    if (!st_view_get(&m->S, name, &v)) return NULL;
    if (v.dtype != 2 || (expect > 0 && v.numel != expect)) {
        fprintf(stderr, "[wh] %s: want f32[%lld], got dtype=%d numel=%lld\n",
                name, (long long)expect, v.dtype, (long long)v.numel);
        exit(1);
    }
    return (const float *)v.p;
}

#define WH_TAU_RES 2048

static void wh_load_cats(WModel *m, const char *snap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/kestrel.json", snap);
    long n = 0;
    char *buf = wh_read_file_(path, &n);
    m->cats_tau = falloc(m->d.layers);
    int level = 0;  /* default off — sparsity is opt-in for max quality */
    const char *e = getenv("WH_SPARSE");
    if (e) level = atoi(e);
    if (level <= 0) { free(buf); return; }
    int loaded = 0;
    if (buf) {
        char *arena = NULL;
        jval *r = json_parse(buf, &arena);
        jval *wh = r ? json_get(r, "windhover") : NULL;
        jval *ct = wh ? json_get(wh, "cats_tau") : NULL;
        if (ct && ct->t == J_ARR && ct->len >= m->d.layers) {
            const char *key = level >= 50 ? "p50" : (level >= 40 ? "p40" : "p25");
            for (int i = 0; i < m->d.layers; i++) {
                jval *t = json_get(ct->kids[i], key);
                m->cats_tau[i] = t ? (float)t->num : 0.f;
            }
            loaded = 1;
            {
                const char *qe = getenv("QUIET");
                if (!(qe && atoi(qe)))
                    fprintf(stderr, "[wh] CATS sparsity on: target %d%% (calibrated taus)\n", level);
            }
        }
        free(arena);
    }
    if (!loaded) {
        /* no offline calibration: collect |act| during prefill, arm taus after */
        m->tau_online = 1;
        m->tau_target_pct = level;
        m->tau_res = falloc((int64_t)m->d.layers * WH_TAU_RES);
        m->tau_n = (int *)balloc((int64_t)m->d.layers * 4);
        {
            const char *qe = getenv("QUIET");
            if (!(qe && atoi(qe)))
                fprintf(stderr, "[wh] CATS sparsity: target %d%% (online calibration "
                        "during prefill)\n", level);
        }
    }
    free(buf);
}

/* reservoir-sample |act| values for layer li (prefill only) */
static void tau_observe(WModel *m, int li, const float *g, int I) {
    float *res = m->tau_res + (int64_t)li * WH_TAU_RES;
    int *np_ = &m->tau_n[li];
    int stride = I / 256 > 0 ? I / 256 : 1;
    for (int j = 0; j < I; j += stride) {
        float a = fabsf(g[j]);
        if (*np_ < WH_TAU_RES) {
            res[(*np_)++] = a;
        } else {
            uint64_t r = (uint64_t)(*np_) * 2654435761u ^ (uint64_t)j * 40503u;
            int k = (int)(r % (uint64_t)(*np_ + 1));
            if (k < WH_TAU_RES) res[k] = a;
            (*np_)++;
        }
    }
}

static int cmp_f32(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

static void tau_arm(WModel *m) {
    if (!m->tau_online) return;
    for (int li = 0; li < m->d.layers; li++) {
        int nres = m->tau_n[li] < WH_TAU_RES ? m->tau_n[li] : WH_TAU_RES;
        if (nres < 64) continue;
        float *res = m->tau_res + (int64_t)li * WH_TAU_RES;
        qsort(res, (size_t)nres, sizeof(float), cmp_f32);
        m->cats_tau[li] = res[(int)((int64_t)nres * m->tau_target_pct / 100)];
    }
    m->tau_online = 0;
}

static void wh_model_init(WModel *m, const char *snap) {
    memset(m, 0, sizeof(*m));
    /* Prefer WMIR dims when present; else config.json allowlist path. */
    if (wmir_load(snap, &m->wmir)) {
        m->d = m->wmir.desc;
    } else if (!wh_desc_from_config(snap, &m->d)) {
        fprintf(stderr, "[wh] unsupported config in %s\n", snap);
        exit(1);
    } else {
        wmir_synthesize_from_desc(&m->d, &m->wmir);
    }
    double t0 = now_s();
    st_init(&m->S, snap);
    WhDesc *d = &m->d;
    int D = d->hidden, I = d->inter, H = d->heads, KV = d->kv_heads, hd = d->head_dim;
    if (KV <= 0 || H % KV != 0) {
        fprintf(stderr, "[wh] invalid heads=%d kv_heads=%d\n", H, KV);
        exit(1);
    }
    if (hd <= 0 || hd % WH_KVG != 0 || hd > WH_MAX_HD) {
        fprintf(stderr, "[wh] head_dim=%d must be in (0,%d] and %% %d == 0\n",
                hd, WH_MAX_HD, WH_KVG);
        exit(1);
    }
    if (D <= 0 || D > WH_MAX_HIDDEN || D % WH_GS != 0) {
        fprintf(stderr, "[wh] hidden=%d must be in (0,%d] and %% %d == 0\n",
                D, WH_MAX_HIDDEN, WH_GS);
        exit(1);
    }

    wt_from_view(m, "model.embed_tokens.weight", &m->embed, d->vocab, D, 1);
    wt_from_view(m, "lm_head.weight", &m->lm, d->vocab, D, 1);
    if (m->embed.fmt == WT_NONE) { fprintf(stderr, "[wh] missing embed\n"); exit(1); }
    if (m->lm.fmt == WT_NONE) m->lm = m->embed;
    m->final_norm = f32_view(m, "model.norm.weight", D);
    if (!m->final_norm) { fprintf(stderr, "[wh] missing final norm\n"); exit(1); }

    m->L = calloc((size_t)d->layers, sizeof(WLayer));
    char nm[320];
    int64_t bytes_full = m->lm.bytes;
    int max_lin = 0;
    for (int i = 0; i < d->layers; i++) {
        WLayer *l = &m->L[i];
        l->kv_share_from = -1;
        l->attn_kind = WMIR_OP_ATTN_GQA;
        l->mlp_kind = (d->act == WH_ACT_GELU_TANH) ? WMIR_OP_MLP_GELU : WMIR_OP_MLP_SWIGLU;
        l->layer_inter = I;
        if (m->wmir.present && i < m->wmir.n_layers) {
            WmirLayer *wl = &m->wmir.layers[i];
            l->attn_kind = wl->attn;
            l->mlp_kind = wl->mlp;
            l->kv_share_from = wl->kv_share_from;
            l->is_sw = wl->is_sw;
            if (wl->inter_override > 0) l->layer_inter = wl->inter_override;
            for (int oi = 0; oi < wl->n_ops; oi++) {
                if (wl->ops[oi].chunk_size > 0) l->chunk_size = wl->ops[oi].chunk_size;
                if (wl->ops[oi].sliding_window > 0) {
                    l->is_sw = 1;
                    if (d->sliding_window <= 0)
                        d->sliding_window = wl->ops[oi].sliding_window;
                }
            }
        }
        #define P(fmtstr) (snprintf(nm, sizeof(nm), fmtstr, i), nm)
        l->in_ln = f32_view(m, P("model.layers.%d.input_layernorm.weight"), D);
        l->post_ln = f32_view(m, P("model.layers.%d.post_attention_layernorm.weight"), D);
        if (d->post_norms) {
            l->pre_ffn_ln = f32_view(m, P("model.layers.%d.pre_feedforward_layernorm.weight"), D);
            l->post_ffn_ln = f32_view(m, P("model.layers.%d.post_feedforward_layernorm.weight"), D);
        }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_norm.weight", i);
        { st_view v; l->q_norm = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_norm.weight", i);
        { st_view v; l->k_norm = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.bias", i);
        { st_view v; l->qb = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.bias", i);
        { st_view v; l->kb = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.bias", i);
        { st_view v; l->vb = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }

        int Li = l->layer_inter > 0 ? l->layer_inter : I;
        if (l->attn_kind == WMIR_OP_ATTN_LINEAR_GDN) {
            /* Linear GDN: optional quantized projections. */
            wt_from_view(m, P("model.layers.%d.linear_attn.in_proj_qkv.weight"),
                         &l->lin_qkv, 0, D, 1);
            if (l->lin_qkv.fmt != WT_NONE) {
                l->lin_dim = l->lin_qkv.O / 3;
                if (l->lin_dim < 1) l->lin_dim = D;
                if (l->lin_dim > max_lin) max_lin = l->lin_dim;
                l->has_linear = 1;
            }
            wt_from_view(m, P("model.layers.%d.linear_attn.out_proj.weight"),
                         &l->lin_out, D, l->lin_dim > 0 ? l->lin_dim : D, 0);
            if (l->lin_out.fmt != WT_NONE && l->lin_dim <= 0) {
                l->lin_dim = l->lin_out.I > 0 ? l->lin_out.I : D;
                if (l->lin_dim > max_lin) max_lin = l->lin_dim;
                l->has_linear = 1;
            }
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.A_log", i);
            { st_view v; l->lin_A_log = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.dt_bias", i);
            { st_view v; l->lin_dt_bias = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
            snprintf(nm, sizeof(nm), "model.layers.%d.linear_attn.norm.weight", i);
            { st_view v; l->lin_norm = st_view_get(&m->S, nm, &v) ? (const float *)v.p : NULL; }
        }

        wt_from_view(m, P("model.layers.%d.self_attn.q_proj.weight"), &l->q, H * hd, D, 1);
        wt_from_view(m, P("model.layers.%d.self_attn.k_proj.weight"), &l->k, KV * hd, D, 1);
        wt_from_view(m, P("model.layers.%d.self_attn.v_proj.weight"), &l->v, KV * hd, D, 1);
        wt_from_view(m, P("model.layers.%d.self_attn.o_proj.weight"), &l->o, D, H * hd, 0);
        l->has_gqa = (l->q.fmt != WT_NONE);
        wt_from_view(m, P("model.layers.%d.mlp.gate_proj.weight"), &l->gate, Li, D, 0);
        wt_from_view(m, P("model.layers.%d.mlp.up_proj.weight"), &l->up, Li, D, 0);
        wt_from_view(m, P("model.layers.%d.mlp.down_proj.weight.t"), &l->downT, Li, D, 0);

        if (l->has_linear && !l->has_gqa) {
            /* Linear-only layer: MLP still required. */
            if (l->gate.fmt == WT_NONE || l->downT.fmt == WT_NONE) {
                fprintf(stderr, "[wh] layer %d linear_gdn missing mlp tensors\n", i);
                exit(1);
            }
        } else if (l->kv_share_from >= 0 && l->has_gqa) {
            /* Shared-KV layer: q+o required; k/v may be absent. */
            if (l->q.fmt == WT_NONE || l->o.fmt == WT_NONE) {
                fprintf(stderr, "[wh] layer %d kv_share missing q/o\n", i);
                exit(1);
            }
            if (l->downT.fmt == WT_NONE) {
                fprintf(stderr, "[wh] %s: missing transposed down_proj — re-convert with "
                        "tools/kestrel_pack.py\n", snap);
                exit(1);
            }
        } else {
            if (l->downT.fmt == WT_NONE) {
                fprintf(stderr, "[wh] %s: missing transposed down_proj — re-convert with "
                        "tools/kestrel_pack.py\n", snap);
                exit(1);
            }
            if (l->downT.fmt != WT_I4G && l->downT.fmt != WT_I8R) {
                fprintf(stderr, "[wh] layer %d downT must be int4-g64 or int8-row (fmt=%d)\n",
                        i, (int)l->downT.fmt);
                exit(1);
            }
            if (l->downT.fmt == WT_I4G && (!l->downT.q4 || !l->downT.sc || !l->downT.zp)) {
                fprintf(stderr, "[wh] layer %d downT int4 missing scales\n", i);
                exit(1);
            }
            if (l->downT.fmt == WT_I8R && (!l->downT.q8 || !l->downT.rs)) {
                fprintf(stderr, "[wh] layer %d downT int8 missing scales\n", i);
                exit(1);
            }
            if (l->q.fmt == WT_NONE || l->gate.fmt == WT_NONE) {
                fprintf(stderr, "[wh] layer %d missing tensors\n", i);
                exit(1);
            }
        }
        if (!m->wmir.present && d->sliding_window > 0)
            l->is_sw = d->sw_pattern > 0 ? ((i % d->sw_pattern) != (d->sw_pattern - 1))
                                         : 1;
        bytes_full += l->q.bytes + l->k.bytes + l->v.bytes + l->o.bytes +
                      l->gate.bytes + l->up.bytes + l->downT.bytes;
        #undef P
    }
    if (max_lin > 0)
        m->lin_state = falloc((int64_t)d->layers * max_lin);
    /* Scratch buffers sized to the widest MLP layer (double-wide, etc.). */
    for (int i = 0; i < d->layers; i++)
        if (m->L[i].layer_inter > d->inter) d->inter = m->L[i].layer_inter;
    m->bytes_full = bytes_full;
    m->load_s = now_s() - t0;
    m->prof = getenv("WH_PROF") ? 1 : 0;
}

/* -------------------------------------------------------------- kv cache */

static void kv_alloc(WModel *m, int max_t) {
    WhDesc *d = &m->d;
    int KV = d->kv_heads, hd = d->head_dim, gs = hd / WH_KVG;
    m->max_t = max_t;
    m->K8 = calloc((size_t)d->layers, sizeof(void *));
    m->V8 = calloc((size_t)d->layers, sizeof(void *));
    m->KS = calloc((size_t)d->layers, sizeof(void *));
    m->VS = calloc((size_t)d->layers, sizeof(void *));
    for (int i = 0; i < d->layers; i++) {
        m->K8[i] = balloc((int64_t)KV * max_t * hd);
        m->V8[i] = balloc((int64_t)KV * max_t * hd);
        m->KS[i] = balloc((int64_t)KV * max_t * gs * 2);
        m->VS[i] = balloc((int64_t)KV * max_t * gs * 2);
    }
}

static inline void kv_store_row(int8_t *dst8, wh_f16 *dsts, const float *src, int hd) {
    for (int g = 0; g < hd; g += WH_KVG) {
        float amax = 0.f;
        for (int j = 0; j < WH_KVG; j++) {
            float a = fabsf(src[g + j]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.f;
        if (s < 1e-12f) s = 1e-12f;
        float inv = 1.f / s;
        dsts[g / WH_KVG] = (wh_f16)s;
        for (int j = 0; j < WH_KVG; j++)
            dst8[g + j] = (int8_t)lrintf(src[g + j] * inv);
    }
}

/* score = qs * sum_g ks[g] * idot32(q8_g, k8_g) */
static inline float kv_score(const int8_t *q8, float qs, const int8_t *k8,
                             const wh_f16 *ks, int hd) {
    float acc = 0.f;
#if defined(__AVX2__)
    for (int g = 0; g < hd; g += WH_KVG)
        acc += (float)ks[g / WH_KVG] * (float)wh_dot_i8i8_avx(q8 + g, k8 + g, WH_KVG);
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int g = 0; g < hd; g += WH_KVG) {
        int32x4_t p = vdupq_n_s32(0);
        p = vdotq_s32(p, vld1q_s8(q8 + g), vld1q_s8(k8 + g));
        p = vdotq_s32(p, vld1q_s8(q8 + g + 16), vld1q_s8(k8 + g + 16));
        acc += (float)ks[g / WH_KVG] * (float)vaddvq_s32(p);
    }
#else
    for (int g = 0; g < hd; g += WH_KVG) {
        int32_t p = 0;
        for (int j = 0; j < WH_KVG; j++) p += (int32_t)q8[g + j] * k8[g + j];
        acc += (float)ks[g / WH_KVG] * (float)p;
    }
#endif
    return acc * qs;
}

static inline void kv_axpy_v(float *ctx, float a, const int8_t *v8,
                             const wh_f16 *vs, int hd) {
#if defined(__AVX2__)
    for (int g = 0; g < hd; g += WH_KVG)
        wh_axpy_i8_avx(ctx + g, v8 + g, a * (float)vs[g / WH_KVG], WH_KVG);
#elif defined(__ARM_NEON)
    for (int g = 0; g < hd; g += WH_KVG) {
        float32x4_t va = vdupq_n_f32(a * (float)vs[g / WH_KVG]);
        for (int j = 0; j < WH_KVG; j += 8) {
            int16x8_t w16 = vmovl_s8(vld1_s8(v8 + g + j));
            float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
            float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
            vst1q_f32(ctx + g + j, vfmaq_f32(vld1q_f32(ctx + g + j), va, lo));
            vst1q_f32(ctx + g + j + 4, vfmaq_f32(vld1q_f32(ctx + g + j + 4), va, hi));
        }
    }
#else
    for (int g = 0; g < hd; g += WH_KVG) {
        float s = a * (float)vs[g / WH_KVG];
        for (int j = 0; j < WH_KVG; j++) ctx[g + j] += s * (float)v8[g + j];
    }
#endif
}

/* ---------------------------------------------------------------- forward */

/* One transformer stack pass over S tokens (positions pos..pos+S-1).
 * Writes logits for the LAST token into m->logit (and, if want_all_logits,
 * for every token into m->logit[s][vocab]). Sparsity only when S==1..4. */
static void wh_forward(WModel *m, const int *tokens, int S, int pos, int want_all_logits) {
    WhDesc *d = &m->d;
    int D = d->hidden, I = d->inter, H = d->heads, KV = d->kv_heads, hd = d->head_dim;
    int gqa = H / KV, ngD = D / WH_GS, ngA = (H * hd) / WH_GS;
    float qscale = d->query_scale > 0 ? d->query_scale : 1.f / sqrtf((float)hd);
    int use_sparse = (S <= 4);

    /* embeddings */
    for (int s = 0; s < S; s++) {
        float *x = m->x + (int64_t)s * D;
        int tokid = tokens[s];
        if (m->embed.fmt == WT_I8R) {
            const int8_t *er = m->embed.q8 + (int64_t)tokid * D;
            float es = m->embed.rs[tokid];
            for (int i = 0; i < D; i++) x[i] = (float)er[i] * es;
        } else {
            memcpy(x, m->embed.f + (int64_t)tokid * D, (size_t)D * sizeof(float));
        }
        if (d->embed_scale > 0)
            for (int i = 0; i < D; i++) x[i] *= d->embed_scale;
    }

    for (int li = 0; li < d->layers; li++) {
        WLayer *l = &m->L[li];
        double tt0 = m->prof ? now_s() : 0;
        int Li = l->layer_inter > 0 ? l->layer_inter : I;
        int kv_src = (l->kv_share_from >= 0 && l->kv_share_from < d->layers)
                         ? l->kv_share_from : li;

        /* ---- attention ---- */
        for (int s = 0; s < S; s++) {
            rmsnorm_row(m->nrm + (int64_t)s * D, m->x + (int64_t)s * D, l->in_ln,
                        D, d->eps, d->norm == WH_NORM_RMS_GEMMA);
            m->sx[s] = qrow(m->nrm + (int64_t)s * D, m->xq + (int64_t)s * D,
                            m->xqsum + (int64_t)s * ngD, D);
        }

        if (l->has_linear && !l->has_gqa) {
            /* Gated DeltaNet-style linear attention (recurrent):
             *   qkv = W x; split q,k,v; state = decay*state + k⊙v; y = q⊙state; out = Wo y
             * Faithful enough for text generation; full GDN conv/delta later. */
            int LD = l->lin_dim > 0 ? l->lin_dim : D;
            for (int s = 0; s < S; s++) {
                float *nrm = m->nrm + (int64_t)s * D;
                float *qkv = m->g + (int64_t)s * I; /* reuse gate scratch before MLP */
                if (3 * LD > I) {
                    fprintf(stderr, "[wh] linear_gdn dim %d exceeds mlp scratch %d\n", LD, I);
                    exit(1);
                }
                memset(qkv, 0, (size_t)(3 * LD) * sizeof(float));
                if (l->lin_qkv.fmt == WT_I8R) {
                    for (int o = 0; o < 3 * LD && o < l->lin_qkv.O; o++) {
                        const int8_t *row = l->lin_qkv.q8 + (int64_t)o * D;
                        float acc = 0.f;
                        for (int j = 0; j < D; j++) acc += (float)row[j] * nrm[j];
                        qkv[o] = acc * l->lin_qkv.rs[o];
                    }
                } else if (l->lin_qkv.fmt == WT_F32 && l->lin_qkv.f) {
                    for (int o = 0; o < 3 * LD && o < l->lin_qkv.O; o++) {
                        const float *row = l->lin_qkv.f + (int64_t)o * D;
                        float acc = 0.f;
                        for (int j = 0; j < D; j++) acc += row[j] * nrm[j];
                        qkv[o] = acc;
                    }
                }
                float *qq = qkv, *kk = qkv + LD, *vv = qkv + 2 * LD;
                float *st = m->lin_state ? m->lin_state + (int64_t)li * LD : NULL;
                float decay = 0.95f;
                if (l->lin_A_log) {
                    float a = l->lin_A_log[0];
                    decay = 1.f / (1.f + expf(-a));
                    if (decay < 0.5f) decay = 0.5f;
                    if (decay > 0.999f) decay = 0.999f;
                }
                float *y = m->ctx + (int64_t)s * H * hd; /* park in ctx then project */
                memset(y, 0, (size_t)D * sizeof(float));
                if (st) {
                    for (int j = 0; j < LD; j++) {
                        st[j] = decay * st[j] + kk[j] * vv[j];
                        y[j % D] += qq[j] * st[j];
                    }
                } else {
                    for (int j = 0; j < LD && j < D; j++) y[j] = qq[j] * kk[j] * vv[j];
                }
                /* out proj into tmp then residual */
                float *out = m->tmp;
                memset(out, 0, (size_t)D * sizeof(float));
                if (l->lin_out.fmt == WT_I8R) {
                    for (int o = 0; o < D; o++) {
                        const int8_t *row = l->lin_out.q8 + (int64_t)o * l->lin_out.I;
                        float acc = 0.f;
                        int nin = l->lin_out.I < LD ? l->lin_out.I : LD;
                        for (int j = 0; j < nin; j++) acc += (float)row[j] * y[j];
                        out[o] = acc * l->lin_out.rs[o];
                    }
                } else if (l->lin_out.fmt == WT_F32 && l->lin_out.f) {
                    for (int o = 0; o < D; o++) {
                        const float *row = l->lin_out.f + (int64_t)o * l->lin_out.I;
                        float acc = 0.f;
                        int nin = l->lin_out.I < LD ? l->lin_out.I : LD;
                        for (int j = 0; j < nin; j++) acc += row[j] * y[j];
                        out[o] = acc;
                    }
                } else {
                    memcpy(out, y, (size_t)D * sizeof(float));
                }
                float *x = m->x + (int64_t)s * D;
                if (d->post_norms && l->post_ln)
                    rmsnorm_row(out, out, l->post_ln, D, d->eps,
                                d->norm == WH_NORM_RMS_GEMMA);
                for (int j = 0; j < D; j++) x[j] += out[j];
            }
            /* fall through to MLP below — skip GQA block */
            goto wh_layer_mlp;
        }

        mm_wt_run(&l->q, m->xq, m->sx, m->xqsum, m->q, S, H * hd);
        if (kv_src == li) {
            mm_wt_run(&l->k, m->xq, m->sx, m->xqsum, m->k, S, KV * hd);
            mm_wt_run(&l->v, m->xq, m->sx, m->xqsum, m->v, S, KV * hd);
        }
        for (int s = 0; s < S; s++) {
            float *q = m->q + (int64_t)s * H * hd;
            float *k = m->k + (int64_t)s * KV * hd;
            float *v = m->v + (int64_t)s * KV * hd;
            int p = pos + s;
            if (l->qb) for (int i = 0; i < H * hd; i++) q[i] += l->qb[i];
            if (kv_src == li) {
                if (l->kb) for (int i = 0; i < KV * hd; i++) k[i] += l->kb[i];
                if (l->vb) for (int i = 0; i < KV * hd; i++) v[i] += l->vb[i];
            }
            if (l->q_norm)
                for (int hh = 0; hh < H; hh++)
                    rmsnorm_row(q + hh * hd, q + hh * hd, l->q_norm, hd, d->eps,
                                d->norm == WH_NORM_RMS_GEMMA);
            if (kv_src == li && l->k_norm)
                for (int hh = 0; hh < KV; hh++)
                    rmsnorm_row(k + hh * hd, k + hh * hd, l->k_norm, hd, d->eps,
                                d->norm == WH_NORM_RMS_GEMMA);
            const float *cs = m->rope_cos + (int64_t)p * m->rope_half;
            const float *sn = m->rope_sin + (int64_t)p * m->rope_half;
            int half = m->rope_half;
            for (int hh = 0; hh < H; hh++) {
                float *qh = q + hh * hd;
                for (int j = 0; j < half; j++) {
                    float a = qh[j], b = qh[j + half];
                    qh[j] = a * cs[j] - b * sn[j];
                    qh[j + half] = b * cs[j] + a * sn[j];
                }
            }
            if (kv_src == li) {
                for (int hh = 0; hh < KV; hh++) {
                    float *kh = k + hh * hd;
                    for (int j = 0; j < half; j++) {
                        float a = kh[j], b = kh[j + half];
                        kh[j] = a * cs[j] - b * sn[j];
                        kh[j + half] = b * cs[j] + a * sn[j];
                    }
                    kv_store_row(m->K8[li] + ((int64_t)hh * m->max_t + p) * hd,
                                 m->KS[li] + ((int64_t)hh * m->max_t + p) * (hd / WH_KVG),
                                 kh, hd);
                    kv_store_row(m->V8[li] + ((int64_t)hh * m->max_t + p) * hd,
                                 m->VS[li] + ((int64_t)hh * m->max_t + p) * (hd / WH_KVG),
                                 v + hh * hd, hd);
                }
            }
        }
        /* attention scores/ctx per (s, head) */
        #pragma omp parallel for schedule(static) collapse(2) if(H * S >= 4)
        for (int s = 0; s < S; s++) {
            for (int hh = 0; hh < H; hh++) {
                int p = pos + s;
                int kvh = hh / gqa;
                int t0 = 0;
                int win = d->sliding_window;
                if (l->attn_kind == WMIR_OP_ATTN_CHUNKED && l->chunk_size > 0)
                    win = l->chunk_size;
                else if (l->attn_kind == WMIR_OP_ATTN_MSA && l->chunk_size > 0)
                    win = l->chunk_size;
                else if (l->attn_kind == WMIR_OP_ATTN_CSA_HCA && win <= 0)
                    win = 128;
                if ((l->is_sw || l->attn_kind == WMIR_OP_ATTN_CHUNKED ||
                     l->attn_kind == WMIR_OP_ATTN_CSA_HCA ||
                     l->attn_kind == WMIR_OP_ATTN_MSA) &&
                    win > 0 && p - win + 1 > 0)
                    t0 = p - win + 1;
                const float *qh = m->q + (int64_t)s * H * hd + hh * hd;
                /* quantize q head (hd validated ≤ WH_MAX_HD at load) */
                int8_t q8[WH_MAX_HD];
                float amax = 0.f;
                for (int j = 0; j < hd; j++) { float a = fabsf(qh[j]); if (a > amax) amax = a; }
                float qs = amax / 127.f;
                if (qs < 1e-12f) qs = 1e-12f;
                float invq = 1.f / qs;
                for (int j = 0; j < hd; j++) q8[j] = (int8_t)lrintf(qh[j] * invq);
                float *sc = m->sc + ((int64_t)s * H + hh) * m->max_t;
                const int8_t *K8 = m->K8[kv_src] + (int64_t)kvh * m->max_t * hd;
                const wh_f16 *KS = m->KS[kv_src] + (int64_t)kvh * m->max_t * (hd / WH_KVG);
                for (int t = t0; t <= p; t++) {
                    float scv = kv_score(q8, qs, K8 + (int64_t)t * hd,
                                         KS + (int64_t)t * (hd / WH_KVG), hd) * qscale;
                    if (d->attn_softcap > 0)
                        scv = d->attn_softcap * tanhf(scv / d->attn_softcap);
                    sc[t] = scv;
                }
                softmax_row(sc + t0, p - t0 + 1);
                float *cx = m->ctx + (int64_t)s * H * hd + hh * hd;
                memset(cx, 0, (size_t)hd * sizeof(float));
                const int8_t *V8 = m->V8[kv_src] + (int64_t)kvh * m->max_t * hd;
                const wh_f16 *VS = m->VS[kv_src] + (int64_t)kvh * m->max_t * (hd / WH_KVG);
                for (int t = t0; t <= p; t++)
                    kv_axpy_v(cx, sc[t], V8 + (int64_t)t * hd,
                              VS + (int64_t)t * (hd / WH_KVG), hd);
            }
        }
        /* o proj */
        for (int s = 0; s < S; s++)
            m->sx[s] = qrow(m->ctx + (int64_t)s * H * hd, m->xq + (int64_t)s * H * hd,
                            m->xqsum + (int64_t)s * ngA, H * hd);
        mm_wt_run(&l->o, m->xq, m->sx, m->xqsum, m->tmp, S, D);
        for (int s = 0; s < S; s++) {
            float *dst = m->x + (int64_t)s * D;
            float *o = m->tmp + (int64_t)s * D;
            if (d->post_norms && l->post_ln) {
                rmsnorm_row(o, o, l->post_ln, D, d->eps, d->norm == WH_NORM_RMS_GEMMA);
            }
            add_inplace(dst, o, D);
        }
        if (m->prof) { m->t_attn += now_s() - tt0; tt0 = now_s(); }

wh_layer_mlp:
        /* ---- mlp ---- */
        if (l->gate.fmt == WT_NONE) {
            if (m->prof) m->t_mlp += now_s() - tt0;
            continue;
        }
        const float *mlp_ln = d->post_norms ? l->pre_ffn_ln : l->post_ln;
        for (int s = 0; s < S; s++) {
            rmsnorm_row(m->nrm + (int64_t)s * D, m->x + (int64_t)s * D, mlp_ln,
                        D, d->eps, d->norm == WH_NORM_RMS_GEMMA);
            m->sx[s] = qrow(m->nrm + (int64_t)s * D, m->xq + (int64_t)s * D,
                            m->xqsum + (int64_t)s * ngD, D);
        }
        mm_wt_run(&l->gate, m->xq, m->sx, m->xqsum, m->g, S, Li);

        float tau = m->cats_tau ? m->cats_tau[li] : 0.f;
        int gelu = d->act == WH_ACT_GELU_TANH;
        if (use_sparse && tau > 0.f) {
            /* CATS: activation, threshold, AU filter, sparse up + down^T */
            for (int s = 0; s < S; s++) {
                float *g = m->g + (int64_t)s * Li;
                act_bulk(g, Li, gelu);
                int nJ = 0;
                for (int j = 0; j < Li; j++) {
                    float a = fabsf(g[j]);
                    if (a > tau) {
                        m->Jlist[nJ] = j;
                        m->Jmag[nJ] = a;
                        nJ++;
                    }
                }
                m->ffn_rows_total += Li;
                int kept = au_filter(&m->au, li, m->Jlist, m->Jmag, nJ,
                                     tau * m->tau_scale_cold, m->Jkeep);
                m->ffn_rows_kept += kept;
                int rowb_up = (l->up.fmt == WT_I4G ? D / 2 + ngD * 4 : D + 4);
                int rowb_dn = (l->downT.fmt == WT_I4G ? D / 2 + ngD * 4 : D + 4);
                au_note_bytes_saved(&m->au, (int64_t)(Li - kept) * (rowb_up + rowb_dn));
                /* compact kept list */
                int nK = 0;
                for (int t = 0; t < nJ; t++)
                    if (m->Jkeep[t]) m->Jlist[nK++] = m->Jlist[t];
                const int8_t *xq = m->xq + (int64_t)s * D;
                const int32_t *xs = m->xqsum + (int64_t)s * ngD;
                float sx = m->sx[s];
                float *out = m->tmp + (int64_t)s * D;
                int nth = m->nthreads;
                #pragma omp parallel
                {
                    int tid = 0;
#ifdef _OPENMP
                    tid = omp_get_thread_num();
#endif
                    float *acc = m->dacc + (int64_t)tid * D;
                    memset(acc, 0, (size_t)D * sizeof(float));
                    #pragma omp for schedule(static) nowait
                    for (int t = 0; t < nK; t++) {
                        int j = m->Jlist[t];
                        float uj, hj;
                        if (l->up.fmt == WT_I4G)
                            uj = dot_i4g(l->up.q4 + (int64_t)j * (D >> 1),
                                         l->up.sc + (int64_t)j * ngD,
                                         l->up.zp + (int64_t)j * ngD, xq, xs, sx, D);
                        else
                            uj = (float)dot_i8i8(l->up.q8 + (int64_t)j * D, xq, D) *
                                 l->up.rs[j] * sx;
                        hj = m->g[(int64_t)s * Li + j] * uj;
                        if (l->downT.fmt == WT_I8R) {
                            axpy_i8_row(acc, l->downT.q8 + (int64_t)j * D,
                                        l->downT.rs[j], hj, D);
                        } else {
                            axpy_i4g_row(acc, l->downT.q4 + (int64_t)j * (D >> 1),
                                         l->downT.sc + (int64_t)j * ngD,
                                         l->downT.zp + (int64_t)j * ngD, hj, D);
                            for (int k = 0; k < l->downT.noc; k++)
                                acc[l->downT.oci[k]] +=
                                    hj * (float)l->downT.oc[(int64_t)j * l->downT.noc + k];
                        }
                    }
                }
                memset(out, 0, (size_t)D * sizeof(float));
                for (int t = 0; t < nth; t++)
                    add_inplace(out, m->dacc + (int64_t)t * D, D);
            }
        } else {
            /* dense: batched up + down^T (weights stream once across S) */
            mm_wt_run(&l->up, m->xq, m->sx, m->xqsum, m->u, S, Li);
            for (int s = 0; s < S; s++) {
                float *g = m->g + (int64_t)s * Li;
                float *u = m->u + (int64_t)s * Li;
                act_bulk(g, Li, gelu);
                if (m->tau_online) tau_observe(m, li, g, Li);
                for (int j = 0; j < Li; j++) g[j] *= u[j];
            }
            int nth = m->nthreads;
            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                float *acc = m->dacc + (int64_t)tid * S * D;
                memset(acc, 0, (size_t)S * D * sizeof(float));
                float dq[WH_MAX_HIDDEN];
                #pragma omp for schedule(static) nowait
                for (int j = 0; j < Li; j++) {
                    if (S == 1) {
                        if (l->downT.fmt == WT_I8R) {
                            axpy_i8_row(acc, l->downT.q8 + (int64_t)j * D,
                                        l->downT.rs[j], m->g[j], D);
                        } else {
                            axpy_i4g_row(acc, l->downT.q4 + (int64_t)j * (D >> 1),
                                         l->downT.sc + (int64_t)j * ngD,
                                         l->downT.zp + (int64_t)j * ngD, m->g[j], D);
                            for (int k = 0; k < l->downT.noc; k++)
                                acc[l->downT.oci[k]] +=
                                    m->g[j] * (float)l->downT.oc[(int64_t)j * l->downT.noc + k];
                        }
                        continue;
                    }
                    if (l->downT.fmt == WT_I8R)
                        deq_i8_row(l->downT.q8 + (int64_t)j * D, l->downT.rs[j], dq, D);
                    else {
                        deq_i4g_row(l->downT.q4 + (int64_t)j * (D >> 1),
                                    l->downT.sc + (int64_t)j * ngD,
                                    l->downT.zp + (int64_t)j * ngD, dq, D);
                        for (int k = 0; k < l->downT.noc; k++)
                            dq[l->downT.oci[k]] +=
                                (float)l->downT.oc[(int64_t)j * l->downT.noc + k];
                    }
                    for (int s = 0; s < S; s++) {
                        float hj = m->g[(int64_t)s * Li + j];
                        if (hj == 0.f) continue;
                        float *as = acc + (int64_t)s * D;
                        int dd = 0;
#if defined(__ARM_NEON)
                        float32x4_t vh = vdupq_n_f32(hj);
                        for (; dd + 4 <= D; dd += 4)
                            vst1q_f32(as + dd, vfmaq_f32(vld1q_f32(as + dd), vh, vld1q_f32(dq + dd)));
#endif
                        for (; dd < D; dd++) as[dd] += hj * dq[dd];
                    }
                }
            }
            for (int s = 0; s < S; s++) {
                float *out = m->tmp + (int64_t)s * D;
                memset(out, 0, (size_t)D * sizeof(float));
                for (int t = 0; t < nth; t++)
                    add_inplace(out, m->dacc + ((int64_t)t * S + s) * D, D);
            }
        }
        for (int s = 0; s < S; s++) {
            float *o = m->tmp + (int64_t)s * D;
            if (d->post_norms && l->post_ffn_ln)
                rmsnorm_row(o, o, l->post_ffn_ln, D, d->eps, 1);
            add_inplace(m->x + (int64_t)s * D, o, D);
        }
        if (m->prof) m->t_mlp += now_s() - tt0;
    }
    m->kv_len = pos + S;

    /* ---- lm head ---- */
    double tl0 = m->prof ? now_s() : 0;
    int s_lo = want_all_logits ? 0 : S - 1;
    for (int s = s_lo; s < S; s++) {
        rmsnorm_row(m->nrm + (int64_t)(s - s_lo) * D, m->x + (int64_t)s * D,
                    m->final_norm, D, d->eps, m->d.norm == WH_NORM_RMS_GEMMA);
        m->sx[s - s_lo] = qrow(m->nrm + (int64_t)(s - s_lo) * D,
                               m->xq + (int64_t)(s - s_lo) * D,
                               m->xqsum + (int64_t)(s - s_lo) * (D / WH_GS), D);
    }
    int SL = S - s_lo;
    mm_wt_run(&m->lm, m->xq, m->sx, m->xqsum, m->logit, SL, m->d.vocab);
    if (d->final_softcap > 0)
        for (int s = 0; s < SL; s++) {
            float *lg = m->logit + (int64_t)s * d->vocab;
            for (int t = 0; t < d->vocab; t++)
                lg[t] = d->final_softcap * tanhf(lg[t] / d->final_softcap);
        }
    if (m->prof) m->t_lm += now_s() - tl0;
}

/* --------------------------------------------------------------- sampling */

static uint64_t g_rng;
static float rngf(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (float)((g_rng >> 11) & 0xFFFFFF) / (float)0x1000000;
}

static int sample_logits(float *logit, int vocab, float temp, int topk,
                         float topp) {
    if (temp <= 0.f) {
        int b = 0; float v = logit[0];
        for (int i = 1; i < vocab; i++) if (logit[i] > v) { v = logit[i]; b = i; }
        return b;
    }
    float inv = 1.f / temp;
    float mx = -1e30f;
    for (int i = 0; i < vocab; i++) if (logit[i] > mx) mx = logit[i];
    double sum = 0;
    for (int i = 0; i < vocab; i++) {
        logit[i] = expf((logit[i] - mx) * inv);
        sum += logit[i];
    }
    /* top-k prune (simple selection over a copy of top values) */
    if (topk > 0 && topk < vocab) {
        /* find k-th largest via partial selection on a heap-less pass */
        float thr = 0.f;
        int found = 0;
        float best = 1e30f;
        for (int r = 0; r < topk; r++) {
            float cur = -1.f;
            for (int i = 0; i < vocab; i++)
                if (logit[i] < best && logit[i] > cur) cur = logit[i];
            best = cur;
            found++;
            (void)found;
        }
        thr = best;
        double s2 = 0;
        for (int i = 0; i < vocab; i++) {
            if (logit[i] < thr) logit[i] = 0.f;
            s2 += logit[i];
        }
        sum = s2;
    }
    if (topp > 0.f && topp < 1.f) {
        /* nucleus: iterative threshold search (few passes, avoids full sort) */
        float lo = 0.f, hi = (float)sum;
        for (int it = 0; it < 24; it++) {
            float mid = 0.5f * (lo + hi);
            double mass = 0;
            for (int i = 0; i < vocab; i++) if (logit[i] >= mid) mass += logit[i];
            if (mass / sum > topp) lo = mid; else hi = mid;
        }
        double s2 = 0;
        for (int i = 0; i < vocab; i++) {
            if (logit[i] < lo) logit[i] = 0.f;
            s2 += logit[i];
        }
        sum = s2;
    }
    double r = rngf() * sum, c = 0;
    for (int i = 0; i < vocab; i++) {
        c += logit[i];
        if (c >= r) return i;
    }
    return vocab - 1;
}

/* ------------------------------------------------------- n-gram speculation */

#define SPEC_NMIN 3
#define SPEC_NMAX 6
#define SPEC_K 8
/* pos = position right after the LATEST occurrence of the n-gram;
 * prev = the occurrence before that. The current suffix always updates pos
 * to "now", so drafting reads prev (the true historical continuation). */
typedef struct { uint64_t key; int pos, prev; } SpecEnt;
typedef struct { SpecEnt *e; int cap; } SpecTab;
static SpecTab g_spec_tab[SPEC_NMAX + 1];

static uint64_t spec_hash(const int *ids, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        h ^= (uint64_t)(uint32_t)ids[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1;
}
static void spec_init(void) {
    for (int n = SPEC_NMIN; n <= SPEC_NMAX; n++) {
        g_spec_tab[n].cap = 1 << 16;
        g_spec_tab[n].e = calloc((size_t)g_spec_tab[n].cap, sizeof(SpecEnt));
    }
}
static void spec_update(const int *ids, int len) {
    for (int n = SPEC_NMIN; n <= SPEC_NMAX; n++) {
        if (len < n) continue;
        uint64_t h = spec_hash(ids + len - n, n);
        SpecTab *t = &g_spec_tab[n];
        SpecEnt *e = &t->e[h & (t->cap - 1)];
        if (e->key == h) {
            if (e->pos != len) { e->prev = e->pos; e->pos = len; }
        } else {
            e->key = h;
            e->pos = len;
            e->prev = 0;
        }
    }
}
static int spec_draft(const int *ids, int len, int *draft, int kmax) {
    for (int n = SPEC_NMAX; n >= SPEC_NMIN; n--) {
        if (len < n) continue;
        uint64_t h = spec_hash(ids + len - n, n);
        SpecTab *t = &g_spec_tab[n];
        SpecEnt *e = &t->e[h & (t->cap - 1)];
        if (e->key != h) continue;
        int src = e->pos < len ? e->pos : e->prev;
        if (src <= 0 || src >= len) continue;
        int k = len - src;
        if (k > kmax) k = kmax;
        int nd = 0;
        for (int j = 0; j < k && src + j < len; j++)
            draft[nd++] = ids[src + j];
        if (nd > 0) return nd;
    }
    return 0;
}

/* ------------------------------------------------------------------- run */

static void wh_alloc_scratch(WModel *m, int max_t) {
    WhDesc *d = &m->d;
    int D = d->hidden, I = d->inter, H = d->heads, hd = d->head_dim;
    int S = WH_MAXS;
    int A = H * hd > D ? H * hd : D;
    if (I > A) A = I;
#ifdef _OPENMP
    m->nthreads = omp_get_max_threads();
#else
    m->nthreads = 1;
#endif
    m->x = falloc((int64_t)S * D);
    m->nrm = falloc((int64_t)S * D);
    m->tmp = falloc((int64_t)S * D);
    m->q = falloc((int64_t)S * H * hd);
    m->k = falloc((int64_t)S * d->kv_heads * hd);
    m->v = falloc((int64_t)S * d->kv_heads * hd);
    m->ctx = falloc((int64_t)S * H * hd);
    m->sc = falloc((int64_t)S * H * max_t);
    m->g = falloc((int64_t)S * I);
    m->u = falloc((int64_t)S * I);
    m->logit = falloc((int64_t)S * d->vocab);
    m->xq = balloc((int64_t)S * A);
    m->sx = falloc(S);
    m->xqsum = (int32_t *)balloc((int64_t)S * (A / WH_GS) * 4);
    m->dacc = falloc((int64_t)m->nthreads * S * D);
    m->Jlist = (int *)balloc((int64_t)I * 4);
    m->Jmag = falloc(I);
    m->Jkeep = (unsigned char *)balloc(I);
    /* Phi-4 / partial RoPE: rotate only rope_dim dims; inv_freq uses /rope_dim */
    {
        int rdim = d->rope_dim > 0 ? d->rope_dim : hd;
        if (rdim < 2) rdim = 2;
        if (rdim > hd) rdim = hd;
        if (rdim & 1) rdim--;
        int half = rdim / 2;
        float attn_scale = d->rope_attn_scale > 0.f ? d->rope_attn_scale : 1.f;
        m->rope_dim = rdim;
        m->rope_half = half;
        m->rope_cos = falloc((int64_t)max_t * half);
        m->rope_sin = falloc((int64_t)max_t * half);
        float *inv_freq = falloc(half);
        for (int j = 0; j < half; j++)
            inv_freq[j] = powf(d->rope_theta, -2.f * j / (float)rdim);
        for (int p = 0; p < max_t; p++)
            for (int j = 0; j < half; j++) {
                float ang = (float)p * inv_freq[j];
                m->rope_cos[(int64_t)p * half + j] = attn_scale * cosf(ang);
                m->rope_sin[(int64_t)p * half + j] = attn_scale * sinf(ang);
            }
        free(inv_freq);
        if (rdim != hd || fabsf(attn_scale - 1.f) > 1e-4f)
            fprintf(stderr,
                    "[wh] RoPE: rotary_dim=%d/%d partial=%.3f attn_scale=%.4f\n",
                    rdim, hd, d->partial_rotary, attn_scale);
    }
}

/* resolve SNAP: prefer <snap>/kpk if it is a KPK pack */
const char *wh_resolve_snap(const char *snap, char *buf, int n) {
    char sub[2048];
    snprintf(sub, sizeof(sub), "%s/kpk", snap);
    if (wh_is_kpk(sub)) {
        snprintf(buf, (size_t)n, "%s", sub);
        return buf;
    }
    if (wh_is_kpk(snap)) {
        snprintf(buf, (size_t)n, "%s", snap);
        return buf;
    }
    return NULL;
}

int wh_can_run(const char *snap) {
    char buf[2048];
    const char *resolved = wh_resolve_snap(snap, buf, sizeof(buf));
    if (!resolved) return 0;
    /* MoE-stream WMIR packs have no dense shards — leave them to engine.c. */
    char meta[2048];
    snprintf(meta, sizeof(meta), "%s/kestrel.json", resolved);
    long n = 0;
    char *raw = wh_read_file_(meta, &n);
    if (!raw) return 1;
    int moe_stream = strstr(raw, "\"moe_stream\"") && strstr(raw, "true");
    int has_shards = strstr(raw, "\"shards\"") && !strstr(raw, "\"shards\": []");
    free(raw);
    if (moe_stream && !has_shards) return 0;
    return 1;
}

int wh_run(int argc, char **argv) {
    (void)argc; (void)argv;
    const char *snap0 = getenv("SNAP");
    if (!snap0) { fprintf(stderr, "SNAP=<dir>\n"); return 1; }
    char snapbuf[2048];
    const char *snap = wh_resolve_snap(snap0, snapbuf, sizeof(snapbuf));
    if (!snap) { fprintf(stderr, "[wh] %s is not a KPK pack\n", snap0); return 1; }

    const char *prompt = getenv("COLI_PROMPT");
    if (!prompt) prompt = getenv("PROMPT");
    if (!prompt) prompt = "Say hello in one short sentence.";
    int ngen = getenv("NGEN") ? atoi(getenv("NGEN")) : 64;
    if (ngen < 1) ngen = 1;
    if (ngen > 4096) ngen = 4096;
    int quiet = getenv("QUIET") ? atoi(getenv("QUIET")) : 0;
    float temp = getenv("TEMP") ? (float)atof(getenv("TEMP")) : 0.f;
    int topk = getenv("TOPK") ? atoi(getenv("TOPK")) : 0;
    float topp = getenv("TOPP") ? (float)atof(getenv("TOPP")) : 0.f;
    if (topp <= 0.f && getenv("NUCLEUS") && temp > 0.f)
        topp = (float)atof(getenv("NUCLEUS"));
    g_rng = getenv("SEED") ? (uint64_t)atoll(getenv("SEED"))
                           : (uint64_t)time(NULL) * 2654435761u ^ (uint64_t)getpid();
    int spec_on = getenv("WH_SPEC") ? atoi(getenv("WH_SPEC")) : 0;
    if (temp > 0.f) spec_on = 0; /* speculation kept greedy-exact only */

    setenv("OMP_WAIT_POLICY", "active", 0);
    setenv("OMP_PROC_BIND", "close", 0);
#if defined(__APPLE__) && defined(_OPENMP)
    /* Default to the P-core count: on 4P+6E the E-cores lower aggregate
     * bandwidth for these kernels (measured: 4T 44 tok/s vs 10T 35). */
    if (!getenv("OMP_NUM_THREADS")) {
        int pc = 0;
        size_t len = sizeof(pc);
        if (sysctlbyname("hw.perflevel0.physicalcpu", &pc, &len, NULL, 0) == 0 && pc > 0)
            omp_set_num_threads(pc);
    }
#endif

    WModel M;
    g_wh = &M;
    wh_model_init(&M, snap);
    WhDesc *d = &M.d;

    char tkp[2048];
    snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    Tok T;
    tok_load(&T, tkp);
    /* Arm every known chat EOS from special tokens + HF generation_config arrays.
     * Missing a family end-marker (e.g. Phi <|end|>) causes runaway decode. */
    int stops[16];
    int nstop = 0;
    wh_arm_chat_stops(&T, d, snap, stops, &nstop, (int)(sizeof(stops) / sizeof(stops[0])));
    if (!quiet && nstop > 0) {
        fprintf(stderr, "[wh] stop tokens:");
        for (int i = 0; i < nstop; i++) fprintf(stderr, " %d", stops[i]);
        fputc('\n', stderr);
    }

    int cap = (int)strlen(prompt) + 64;
    if (cap < 256) cap = 256;
    int *ids = malloc((size_t)(cap + ngen + SPEC_K + 8) * sizeof(int));
    int np = tok_encode(&T, prompt, (int)strlen(prompt), ids, cap);
    if (np < 1) { fprintf(stderr, "[wh] empty prompt after tokenization\n"); return 1; }

    int ctx_env = getenv("CTX") ? atoi(getenv("CTX")) : 0;
    int max_t = np + ngen + SPEC_K + 8;
    if (ctx_env > max_t) max_t = ctx_env;
    if (d->max_position > 0 && max_t > d->max_position) max_t = d->max_position;

    /* Prefill writes KV at every prompt position — never allow np > max_t. */
    if (np > max_t) {
        fprintf(stderr, "[wh] prompt tokens (%d) exceed context (%d); truncating\n",
                np, max_t);
        np = max_t;
        if (ngen > 0 && np >= max_t) {
            /* leave at least one decode slot when possible */
            int keep = max_t - 1;
            if (keep < 1) keep = 1;
            np = keep;
        }
    }
    if (np + ngen > max_t) ngen = max_t - np;
    if (ngen < 1) {
        fprintf(stderr, "[wh] no room for decode after prompt (np=%d max_t=%d)\n",
                np, max_t);
        return 1;
    }

    kv_alloc(&M, max_t);
    wh_alloc_scratch(&M, max_t);
    wh_load_cats(&M, snap);
    M.tau_scale_cold = 2.0f;

    /* ---- budget: ledger + AU plan + optional hard cap ---- */
    double ram_gb = getenv("RAM_GB") ? atof(getenv("RAM_GB")) : 0.0;
    int64_t budget = ram_gb > 0 ? (int64_t)(ram_gb * 1e9) : 0;
    {
        int D_ = d->hidden, ngD = D_ / WH_GS;
        int64_t unit = (int64_t)AU_BUNDLE *
            ((D_ / 2 + ngD * 4) * 2 +                 /* up + downT rows */
             (D_ / 2 + ngD * 4));                     /* gate row */
        int64_t ffn = (int64_t)d->layers * ((M.L[0].gate.bytes) + M.L[0].up.bytes +
                                            M.L[0].downT.bytes);
        int64_t kv_bytes = (int64_t)d->layers * d->kv_heads * max_t *
                           (2 * d->head_dim + 4 * (d->head_dim / WH_KVG));
        int64_t other = M.bytes_full - ffn + kv_bytes + (int64_t)6e8;
        au_plan(&M.au, d->layers, d->inter, unit, budget, other);
        au_ledger_set_budget(budget);
        if (budget > 0) budget_apply_hard_cap(budget);
        /* mlock hot prefix + advise cold tail */
        int do_mlock = getenv("MLOCK") ? atoi(getenv("MLOCK")) : (M.au.enabled ? 1 : 0);
        if (M.au.enabled) {
            for (int i = 0; i < d->layers; i++) {
                WLayer *l = &M.L[i];
                int64_t hot_rows = (int64_t)M.au.hot_units * AU_BUNDLE;
                int64_t up_pre = hot_rows * (D_ / 2);
                st_view hv;
                hv.p = l->up.q4; hv.nbytes = up_pre;
                if (do_mlock) st_view_lock(&hv, 1); else st_view_advise(&hv, 1);
                hv.p = l->downT.q4; hv.nbytes = hot_rows * (D_ / 2);
                if (do_mlock) st_view_lock(&hv, 1); else st_view_advise(&hv, 1);
                hv.p = l->up.q4 + up_pre;
                hv.nbytes = ((int64_t)d->inter - hot_rows) * (D_ / 2);
                if (hv.nbytes > 0) st_view_advise(&hv, 0);
            }
        }
    }

    if (!quiet)
        fprintf(stderr, "[wh] %s | %s L=%d D=%d I=%d | %.2f GB pack | load %.2fs | "
                "footprint %.2f GB\n",
                snap, d->model_type, d->layers, d->hidden, d->inter,
                M.bytes_full / 1e9, M.load_s, footprint_gb());

    /* ---- prefill in chunks ---- */
    double tp0 = now_s();
    int pos = 0;
    while (pos < np) {
        int S = np - pos;
        if (S > WH_PREFILL_S) S = WH_PREFILL_S;
        wh_forward(&M, ids + pos, S, pos, 0);
        pos += S;
    }
    double prefill_s = now_s() - tp0;
    tau_arm(&M);   /* online CATS thresholds from prefill activations */
    if (getenv("WH_LOGITS")) {   /* parity harness: dump prefill logits */
        FILE *lf = fopen(getenv("WH_LOGITS"), "wb");
        if (lf) {
            fwrite(M.logit, sizeof(float), (size_t)d->vocab, lf);
            fclose(lf);
        }
    }
    if (!quiet)
        fprintf(stderr, "[wh] prefill %d tok in %.2fs (%.1f tok/s)\n",
                np, prefill_s, prefill_s > 0 ? np / prefill_s : 0);

    /* ---- decode ---- */
    if (spec_on) spec_init();
    if (spec_on)
        for (int i = SPEC_NMIN; i <= np; i++) spec_update(ids, i);

    M.t_attn = M.t_mlp = M.t_lm = 0;
    double t0 = now_s();
    int generated = 0, steps = 0, accepted_total = 0;
    char outbuf[8192];
    int cur_len = np;
    float *lg = M.logit;
    int next = sample_logits(lg, d->vocab, temp, topk, topp);
    while (generated < ngen) {
        ids[cur_len++] = next;
        generated++;
        if (spec_on) spec_update(ids, cur_len);
        /* Never emit stop/EOS control tokens (e.g. <|end|>, <|im_end|>) into chat text. */
        if (wh_is_stop(stops, nstop, next)) break;
        int nch = tok_decode(&T, &next, 1, outbuf, (int)sizeof(outbuf) - 1);
        if (nch > 0) { outbuf[nch] = 0; fputs(outbuf, stdout); fflush(stdout); }
        if (generated >= ngen) break;
        if (cur_len + SPEC_K + 1 >= max_t) break;

        int draft[SPEC_K];
        int nd = spec_on ? spec_draft(ids, cur_len, draft, SPEC_K) : 0;
        if (nd > 0) {
            /* verify batch: [next_input, draft...] -> logits for all */
            int batch[SPEC_K + 1];
            batch[0] = ids[cur_len - 1];
            for (int j = 0; j < nd; j++) batch[j + 1] = draft[j];
            wh_forward(&M, batch, nd + 1, cur_len - 1, 1);
            steps++;
            int acc = 0;
            for (int j = 0; j < nd; j++) {
                float *lj = M.logit + (int64_t)j * d->vocab;
                int am = 0; float v = lj[0];
                for (int i2 = 1; i2 < d->vocab; i2++)
                    if (lj[i2] > v) { v = lj[i2]; am = i2; }
                if (am == draft[j]) {
                    ids[cur_len++] = am;
                    generated++;
                    accepted_total++;
                    acc++;
                    if (spec_on) spec_update(ids, cur_len);
                    if (wh_is_stop(stops, nstop, am) || generated >= ngen) break;
                    int nc2 = tok_decode(&T, &am, 1, outbuf, (int)sizeof(outbuf) - 1);
                    if (nc2 > 0) { outbuf[nc2] = 0; fputs(outbuf, stdout); fflush(stdout); }
                } else break;
            }
            /* rewind kv to cur_len (we computed nd+1 positions from cur_len-1) */
            M.kv_len = cur_len;
            float *last = M.logit + (int64_t)acc * d->vocab;
            next = sample_logits(last, d->vocab, temp, topk, topp);
            if (wh_is_stop(stops, nstop, ids[cur_len - 1])) break;
        } else {
            wh_forward(&M, &ids[cur_len - 1], 1, cur_len - 1, 0);
            steps++;
            next = sample_logits(M.logit, d->vocab, temp, topk, topp);
        }
    }
    fputc('\n', stdout);
    fflush(stdout);
    double dt = now_s() - t0;
    double tps = dt > 0 ? generated / dt : 0;

    if (getenv("WH_STATS") || !quiet) {
        char aub[256], ledb[256];
        au_stats_line(&M.au, aub, sizeof(aub));
        double sp = M.ffn_rows_total
                    ? 100.0 * (1.0 - (double)M.ffn_rows_kept / M.ffn_rows_total) : 0;
        int64_t ffn_b = (M.L[0].up.bytes + M.L[0].downT.bytes) * d->layers;
        double bytes_tok = (M.bytes_full - (double)ffn_b * sp / 100.0) / 1e9;
        au_ledger_report("weights", M.bytes_full, 0, 0);
        au_ledger_line(ledb, sizeof(ledb));
        fprintf(stderr,
                "[wh] decode %.2f tok/s (%d tok, %d fwd) | prefill %.1f tok/s | "
                "RSS %.2f GB | footprint %.2f GB\n"
                "[wh] sparsity %.1f%% | ~%.2f GB/tok | %s | %s\n",
                tps, generated, steps, prefill_s > 0 ? np / prefill_s : 0,
                rss_gb(), footprint_gb(), sp, bytes_tok, aub, ledb);
        if (spec_on)
            fprintf(stderr, "[wh] spec: %d accepted / %d generated (%.2f tok/fwd)\n",
                    accepted_total, generated,
                    steps > 0 ? (double)generated / steps : 0);
        if (M.prof)
            fprintf(stderr, "[wh][prof] attn=%.2fs mlp=%.2fs lm=%.2fs\n",
                    M.t_attn, M.t_mlp, M.t_lm);
    }
    if (getenv("WH_JSON_STATS")) {
        double sp = M.ffn_rows_total
                    ? 100.0 * (1.0 - (double)M.ffn_rows_kept / M.ffn_rows_total) : 0;
        int64_t ffn_b = (M.L[0].up.bytes + M.L[0].downT.bytes) * d->layers;
        double bytes_tok = (M.bytes_full - (double)ffn_b * sp / 100.0);
        int64_t au_touch = M.au.hot_hits + M.au.cold_fetches + M.au.cold_drops;
        double hit = au_touch
            ? 100.0 * (double)M.au.hot_hits / (double)au_touch : 100.0;
        printf("\n@@WH_STATS@@{\"decode_tok_s\":%.2f,\"prefill_tok_s\":%.1f,"
               "\"rss_gb\":%.2f,\"footprint_gb\":%.2f,\"sparsity_pct\":%.1f,"
               "\"bytes_per_tok\":%.0f,\"au_hit_pct\":%.1f,"
               "\"tokens\":%d,\"forwards\":%d}\n",
               tps, prefill_s > 0 ? np / prefill_s : 0, rss_gb(), footprint_gb(),
               sp, bytes_tok, hit, generated, steps);
    }
    free(ids);
    g_wh = NULL;
    return 0;
}
