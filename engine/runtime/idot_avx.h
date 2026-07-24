/* Shared AVX2 / AVX-VNNI int8 and int4 IDOT helpers for dense + KPK paths.
 * Ported from engine.c so Windows/x86 is not stuck on scalar kernels. */
#ifndef WINDHOVER_IDOT_AVX_H
#define WINDHOVER_IDOT_AVX_H

#if defined(__AVX2__)
#include <immintrin.h>

static inline int wh_hsum256_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v), hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}

static inline float wh_hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

#if defined(__AVXVNNI__)
static inline int wh_hsum128_i32(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(v);
}
#endif

/* int8·int8: sign trick (|w| unsigned × x·sign(w)). */
static inline int32_t wh_dot_i8i8_avx(const int8_t *w, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
#if defined(__AVXVNNI__)
    __m128i acc = _mm_setzero_si128();
    for (; i + 16 <= I; i += 16) {
        __m128i wv = _mm_loadu_si128((const __m128i *)(w + i));
        __m128i xv = _mm_loadu_si128((const __m128i *)(x + i));
        __m128i xs = _mm_sign_epi8(xv, wv);
        acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(wv), xs);
    }
    sum = wh_hsum128_i32(acc);
#else
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (; i + 32 <= I; i += 32) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(w + i));
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i p = _mm256_maddubs_epi16(
            _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    sum = wh_hsum256_i32(acc);
#endif
    for (; i < I; i++) sum += (int32_t)w[i] * x[i];
    return sum;
}

/* Packed int4 (nibbles) · int8 over I elements (I even). Returns raw int product. */
static inline int32_t wh_dot_i4i8_avx(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int i = 0;
#if defined(__AVXVNNI__)
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    __m128i acc = _mm_setzero_si128();
    for (; i + 32 <= I; i += 32) {
        __m128i by = _mm_loadu_si128((const __m128i *)(w4 + (i >> 1)));
        __m128i lo = _mm_and_si128(by, m4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
        __m128i n0 = _mm_unpacklo_epi8(lo, hi), n1 = _mm_unpackhi_epi8(lo, hi);
        __m128i w0 = _mm_sub_epi8(n0, b8), w1 = _mm_sub_epi8(n1, b8);
        __m128i x0 = _mm_loadu_si128((const __m128i *)(x + i));
        __m128i x1 = _mm_loadu_si128((const __m128i *)(x + i + 16));
        acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(w0), _mm_sign_epi8(x0, w0));
        acc = _mm_dpbusd_epi32(acc, _mm_abs_epi8(w1), _mm_sign_epi8(x1, w1));
    }
    sum = wh_hsum128_i32(acc);
#else
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256i b8 = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= I; i += 32) {
        __m128i by = _mm_loadu_si128((const __m128i *)(w4 + (i >> 1)));
        __m128i lo = _mm_and_si128(by, m4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
        __m128i n0 = _mm_unpacklo_epi8(lo, hi), n1 = _mm_unpackhi_epi8(lo, hi);
        __m256i wv = _mm256_sub_epi8(_mm256_set_m128i(n1, n0), b8);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
        __m256i p = _mm256_maddubs_epi16(
            _mm256_sign_epi8(wv, wv), _mm256_sign_epi8(xv, wv));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    sum = wh_hsum256_i32(acc);
#endif
    for (; i < I; i += 2) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i] + ((int)(b >> 4) - 8) * x[i + 1];
    }
    return sum;
}

/* Exact f32 · int8 row dot (QKV path). */
static inline float wh_dot_f32_i8_avx(const float *x, const int8_t *w, int I) {
    int i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= I; i += 8) {
        __m256i wi = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i)));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_cvtepi32_ps(wi), acc);
    }
    float a = wh_hsum256_ps(acc);
    for (; i < I; i++) a += x[i] * (float)w[i];
    return a;
}

/* acc[0..32) += vs*q + vz for one packed 16-byte int4 chunk (32 nibbles). */
static inline void wh_axpy_i4g32_avx(float *o, const uint8_t *wg16,
                                     float vs, float vz) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    __m256 vvs = _mm256_set1_ps(vs), vvz = _mm256_set1_ps(vz);
    __m128i by = _mm_loadu_si128((const __m128i *)wg16);
    __m128i lo = _mm_and_si128(by, m4);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
    __m128i n0 = _mm_unpacklo_epi8(lo, hi);
    __m128i n1 = _mm_unpackhi_epi8(lo, hi);
    __m128i q0 = _mm_sub_epi8(n0, b8);
    __m128i q1 = _mm_sub_epi8(n1, b8);
    for (int part = 0; part < 2; part++) {
        __m128i q = part ? q1 : q0;
        __m256i i0 = _mm256_cvtepi8_epi32(q);
        __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q, 8));
        float *p = o + part * 16;
        __m256 a0 = _mm256_loadu_ps(p);
        __m256 a1 = _mm256_loadu_ps(p + 8);
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), vvs, _mm256_add_ps(a0, vvz));
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), vvs, _mm256_add_ps(a1, vvz));
        _mm256_storeu_ps(p, a0);
        _mm256_storeu_ps(p + 8, a1);
    }
}

/* acc[i] += s * q8[i] for I elements. */
static inline void wh_axpy_i8_avx(float *acc, const int8_t *q8, float s, int I) {
    int i = 0;
    __m256 vs = _mm256_set1_ps(s);
    for (; i + 16 <= I; i += 16) {
        __m128i q = _mm_loadu_si128((const __m128i *)(q8 + i));
        __m256i i0 = _mm256_cvtepi8_epi32(q);
        __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q, 8));
        __m256 a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), vs, _mm256_loadu_ps(acc + i));
        __m256 a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), vs, _mm256_loadu_ps(acc + i + 8));
        _mm256_storeu_ps(acc + i, a0);
        _mm256_storeu_ps(acc + i + 8, a1);
    }
    for (; i < I; i++) acc[i] += s * (float)q8[i];
}
#endif /* __AVX2__ */

#endif /* WINDHOVER_IDOT_AVX_H */
