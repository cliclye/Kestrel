/* wmir.h — Windhover Model IR (WMIR)
 *
 * Architecture-agnostic layer graph stored under kestrel.json → windhover.wmir.
 * Packers lower HF configs into WMIR; the dense KPK runtime executes ops by
 * kind instead of hard-coding model_type allowlists.
 *
 * Schema version 1 keys (under windhover.wmir):
 *   version, text_only, family, model{…}, layers[{ops:[…]}], required_ops[]
 *
 * Op kinds (string):
 *   rms_norm | rms_norm_gemma
 *   attn_gqa | attn_mla | attn_linear_gdn | attn_chunked | attn_csa_hca | attn_msa
 *   mlp_swiglu | mlp_gelu | mlp_double_wide
 *   moe_routed | kv_share | embed | lm_head | logit_softcap | ple_gate
 */
#ifndef WH_WMIR_H
#define WH_WMIR_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "json.h"
#include "model_desc.h"

#define WMIR_MAX_LAYERS 256
#define WMIR_MAX_OPS_PER_LAYER 16
#define WMIR_MAX_OP_NAME 32
#define WMIR_MAX_FAMILY 64

typedef enum {
    WMIR_OP_NONE = 0,
    WMIR_OP_RMS_NORM,
    WMIR_OP_RMS_NORM_GEMMA,
    WMIR_OP_ATTN_GQA,
    WMIR_OP_ATTN_MLA,
    WMIR_OP_ATTN_LINEAR_GDN,
    WMIR_OP_ATTN_CHUNKED,
    WMIR_OP_ATTN_CSA_HCA,
    WMIR_OP_ATTN_MSA,
    WMIR_OP_MLP_SWIGLU,
    WMIR_OP_MLP_GELU,
    WMIR_OP_MLP_DOUBLE_WIDE,
    WMIR_OP_MOE_ROUTED,
    WMIR_OP_KV_SHARE,
    WMIR_OP_EMBED,
    WMIR_OP_LM_HEAD,
    WMIR_OP_LOGIT_SOFTCAP,
    WMIR_OP_PLE_GATE,
    WMIR_OP_UNKNOWN
} WmirOpKind;

typedef struct {
    WmirOpKind kind;
    char name[WMIR_MAX_OP_NAME];
    /* Optional op params (0 = unset / default). */
    int sliding_window;
    int kv_share_from;   /* layer index to reuse KV; -1 = none */
    int chunk_size;
    int top_k;
    int n_experts;
    int shared_expert;
    int inter;           /* per-layer MLP width override */
    int head_dim;        /* per-layer head_dim override */
    float softcap;
    char router[16];     /* softmax | sigmoid | hash */
} WmirOp;

typedef struct {
    int n_ops;
    WmirOp ops[WMIR_MAX_OPS_PER_LAYER];
    WmirOpKind attn;     /* primary attention op (resolved) */
    WmirOpKind mlp;      /* primary mlp / moe op */
    int is_sw;
    int kv_share_from;   /* -1 none */
    int double_wide;
    int inter_override;
    int head_dim_override;
    int moe_top_k;
    int moe_n_experts;
} WmirLayer;

typedef struct {
    int present;         /* 1 if windhover.wmir loaded */
    int version;
    int text_only;
    char family[WMIR_MAX_FAMILY];
    int n_layers;
    WmirLayer layers[WMIR_MAX_LAYERS];
    int n_required;
    char required_ops[32][WMIR_MAX_OP_NAME];
    /* Model dims mirrored into WhDesc when loading. */
    WhDesc desc;
} WmirGraph;

static WmirOpKind wmir_op_parse(const char *s) {
    if (!s) return WMIR_OP_NONE;
    if (!strcmp(s, "rms_norm")) return WMIR_OP_RMS_NORM;
    if (!strcmp(s, "rms_norm_gemma")) return WMIR_OP_RMS_NORM_GEMMA;
    if (!strcmp(s, "attn_gqa")) return WMIR_OP_ATTN_GQA;
    if (!strcmp(s, "attn_mla")) return WMIR_OP_ATTN_MLA;
    if (!strcmp(s, "attn_linear_gdn")) return WMIR_OP_ATTN_LINEAR_GDN;
    if (!strcmp(s, "attn_chunked")) return WMIR_OP_ATTN_CHUNKED;
    if (!strcmp(s, "attn_csa_hca")) return WMIR_OP_ATTN_CSA_HCA;
    if (!strcmp(s, "attn_msa")) return WMIR_OP_ATTN_MSA;
    if (!strcmp(s, "mlp_swiglu")) return WMIR_OP_MLP_SWIGLU;
    if (!strcmp(s, "mlp_gelu")) return WMIR_OP_MLP_GELU;
    if (!strcmp(s, "mlp_double_wide")) return WMIR_OP_MLP_DOUBLE_WIDE;
    if (!strcmp(s, "moe_routed")) return WMIR_OP_MOE_ROUTED;
    if (!strcmp(s, "kv_share")) return WMIR_OP_KV_SHARE;
    if (!strcmp(s, "embed")) return WMIR_OP_EMBED;
    if (!strcmp(s, "lm_head")) return WMIR_OP_LM_HEAD;
    if (!strcmp(s, "logit_softcap")) return WMIR_OP_LOGIT_SOFTCAP;
    if (!strcmp(s, "ple_gate")) return WMIR_OP_PLE_GATE;
    return WMIR_OP_UNKNOWN;
}

static int wmir_op_is_attn(WmirOpKind k) {
    return k == WMIR_OP_ATTN_GQA || k == WMIR_OP_ATTN_MLA ||
           k == WMIR_OP_ATTN_LINEAR_GDN || k == WMIR_OP_ATTN_CHUNKED ||
           k == WMIR_OP_ATTN_CSA_HCA || k == WMIR_OP_ATTN_MSA;
}
static int wmir_op_is_mlp(WmirOpKind k) {
    return k == WMIR_OP_MLP_SWIGLU || k == WMIR_OP_MLP_GELU ||
           k == WMIR_OP_MLP_DOUBLE_WIDE || k == WMIR_OP_MOE_ROUTED;
}

/* Kernel registry — ops the linked binary can execute today. */
static int wmir_kernel_supported(WmirOpKind k) {
    switch (k) {
    case WMIR_OP_NONE:
    case WMIR_OP_RMS_NORM:
    case WMIR_OP_RMS_NORM_GEMMA:
    case WMIR_OP_ATTN_GQA:
    case WMIR_OP_ATTN_CHUNKED:      /* chunked → causal GQA + chunk window */
    case WMIR_OP_ATTN_CSA_HCA:      /* CSA/HCA → sliding GQA approximation */
    case WMIR_OP_ATTN_MSA:          /* MSA → block-sparse GQA approximation */
    case WMIR_OP_ATTN_LINEAR_GDN:   /* gated linear recurrence */
    case WMIR_OP_ATTN_MLA:          /* MLA dense path uses MoE engine; flagged */
    case WMIR_OP_MLP_SWIGLU:
    case WMIR_OP_MLP_GELU:
    case WMIR_OP_MLP_DOUBLE_WIDE:
    case WMIR_OP_MOE_ROUTED:
    case WMIR_OP_KV_SHARE:
    case WMIR_OP_EMBED:
    case WMIR_OP_LM_HEAD:
    case WMIR_OP_LOGIT_SOFTCAP:
    case WMIR_OP_PLE_GATE:          /* optional; skipped if tensors absent */
        return 1;
    default:
        return 0;
    }
}

static int wmir_ji(jval *o, const char *k, int def) {
    if (!o) return def;
    jval *v = json_get(o, k);
    if (!v) return def;
    if (v->t == J_ARR && v->len > 0) v = v->kids[0];
    return (int)v->num;
}
static float wmir_jf(jval *o, const char *k, float def) {
    if (!o) return def;
    jval *v = json_get(o, k);
    return v ? (float)v->num : def;
}
static void wmir_js(jval *o, const char *k, char *dst, int n) {
    dst[0] = 0;
    if (!o) return;
    jval *v = json_get(o, k);
    if (v && v->t == J_STR) snprintf(dst, (size_t)n, "%s", v->str);
}

static void wmir_resolve_layer(WmirLayer *L) {
    L->attn = WMIR_OP_ATTN_GQA;
    L->mlp = WMIR_OP_MLP_SWIGLU;
    L->kv_share_from = -1;
    L->double_wide = 0;
    L->inter_override = 0;
    L->head_dim_override = 0;
    L->moe_top_k = 0;
    L->moe_n_experts = 0;
    for (int i = 0; i < L->n_ops; i++) {
        WmirOp *op = &L->ops[i];
        if (wmir_op_is_attn(op->kind)) {
            L->attn = op->kind;
            if (op->sliding_window > 0) L->is_sw = 1;
            if (op->head_dim > 0) L->head_dim_override = op->head_dim;
        }
        if (wmir_op_is_mlp(op->kind)) {
            L->mlp = op->kind;
            if (op->kind == WMIR_OP_MLP_DOUBLE_WIDE) L->double_wide = 1;
            if (op->inter > 0) L->inter_override = op->inter;
            if (op->kind == WMIR_OP_MOE_ROUTED) {
                L->moe_top_k = op->top_k > 0 ? op->top_k : 2;
                L->moe_n_experts = op->n_experts;
            }
        }
        if (op->kind == WMIR_OP_KV_SHARE)
            L->kv_share_from = op->kv_share_from;
    }
}

/* Fill WhDesc defaults from wmir.model object. */
static void wmir_desc_from_model(jval *model, WhDesc *d) {
    memset(d, 0, sizeof(*d));
    wmir_js(model, "model_type", d->model_type, (int)sizeof(d->model_type));
    d->hidden = wmir_ji(model, "hidden", 0);
    d->layers = wmir_ji(model, "layers", 0);
    d->heads = wmir_ji(model, "heads", 0);
    d->kv_heads = wmir_ji(model, "kv_heads", d->heads);
    d->head_dim = wmir_ji(model, "head_dim", d->heads ? d->hidden / d->heads : 0);
    d->inter = wmir_ji(model, "inter", 0);
    d->vocab = wmir_ji(model, "vocab", 0);
    d->rope_theta = wmir_jf(model, "rope_theta", 10000.f);
    d->eps = wmir_jf(model, "eps", 1e-6f);
    d->eos_id = wmir_ji(model, "eos", -1);
    d->bos_id = wmir_ji(model, "bos", -1);
    d->tie_embeddings = wmir_ji(model, "tie_embeddings", 0);
    d->max_position = wmir_ji(model, "max_position", 0);
    d->sliding_window = wmir_ji(model, "sliding_window", 0);
    d->sw_pattern = wmir_ji(model, "sw_pattern", 0);
    d->qkv_bias = wmir_ji(model, "qkv_bias", 0);
    d->qk_norm = wmir_ji(model, "qk_norm", 0);
    d->post_norms = wmir_ji(model, "post_norms", 0);
    d->attn_softcap = wmir_jf(model, "attn_softcap", 0.f);
    d->final_softcap = wmir_jf(model, "final_softcap", 0.f);
    d->embed_scale = wmir_jf(model, "embed_scale", 0.f);
    d->query_scale = wmir_jf(model, "query_scale", 0.f);
    d->partial_rotary = wmir_jf(model, "partial_rotary", 1.f);
    d->rope_attn_scale = wmir_jf(model, "rope_attn_scale", 1.f);
    d->rope_dim = d->head_dim;
    if (d->partial_rotary < 1.f - 1e-6f && d->head_dim > 0) {
        int rd = (int)(d->head_dim * d->partial_rotary);
        if (rd < 2) rd = 2;
        if (rd > d->head_dim) rd = d->head_dim;
        if (rd & 1) rd--;
        d->rope_dim = rd;
    }
    char act[32], norm[32];
    wmir_js(model, "act", act, (int)sizeof(act));
    wmir_js(model, "norm", norm, (int)sizeof(norm));
    d->act = (!strcmp(act, "gelu_tanh") || !strcmp(act, "gelu"))
                 ? WH_ACT_GELU_TANH : WH_ACT_SILU;
    d->norm = (!strcmp(norm, "rmsnorm_gemma") || !strcmp(norm, "gemma"))
                  ? WH_NORM_RMS_GEMMA : WH_NORM_RMS;
}

static int wmir_parse_layer(jval *layer, WmirLayer *L) {
    memset(L, 0, sizeof(*L));
    L->kv_share_from = -1;
    jval *ops = json_get(layer, "ops");
    if (!ops || ops->t != J_ARR) return 0;
    for (int i = 0; i < ops->len && L->n_ops < WMIR_MAX_OPS_PER_LAYER; i++) {
        jval *o = ops->kids[i];
        if (!o || o->t != J_OBJ) continue;
        WmirOp *op = &L->ops[L->n_ops];
        memset(op, 0, sizeof(*op));
        op->kv_share_from = -1;
        wmir_js(o, "op", op->name, WMIR_MAX_OP_NAME);
        op->kind = wmir_op_parse(op->name);
        op->sliding_window = wmir_ji(o, "sliding_window", 0);
        op->kv_share_from = wmir_ji(o, "kv_share_from", -1);
        op->chunk_size = wmir_ji(o, "chunk_size", 0);
        op->top_k = wmir_ji(o, "top_k", 0);
        op->n_experts = wmir_ji(o, "n_experts", 0);
        op->shared_expert = wmir_ji(o, "shared_expert", 0);
        op->inter = wmir_ji(o, "inter", 0);
        op->head_dim = wmir_ji(o, "head_dim", 0);
        op->softcap = wmir_jf(o, "softcap", 0.f);
        wmir_js(o, "router", op->router, (int)sizeof(op->router));
        L->n_ops++;
    }
    wmir_resolve_layer(L);
    return L->n_ops > 0;
}

/* Load WMIR from SNAP/kestrel.json → windhover.wmir. Returns 1 on success. */
static int wmir_load(const char *snap, WmirGraph *g) {
    memset(g, 0, sizeof(*g));
    char path[2048];
    snprintf(path, sizeof(path), "%s/kestrel.json", snap);
    long n = 0;
    char *buf = wh_read_file_(path, &n);
    if (!buf) return 0;
    if (!strstr(buf, "\"wmir\"")) { free(buf); return 0; }
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    if (!root) { free(buf); return 0; }
    jval *wh = json_get(root, "windhover");
    if (!wh) { free(buf); free(arena); return 0; }
    jval *wmir = json_get(wh, "wmir");
    if (!wmir || wmir->t != J_OBJ) { free(buf); free(arena); return 0; }

    g->present = 1;
    g->version = wmir_ji(wmir, "version", 1);
    g->text_only = wmir_ji(wmir, "text_only", 1);
    wmir_js(wmir, "family", g->family, WMIR_MAX_FAMILY);

    jval *model = json_get(wmir, "model");
    if (model && model->t == J_OBJ) wmir_desc_from_model(model, &g->desc);

    jval *req = json_get(wmir, "required_ops");
    if (req && req->t == J_ARR) {
        for (int i = 0; i < req->len && g->n_required < 32; i++) {
            if (req->kids[i] && req->kids[i]->t == J_STR)
                snprintf(g->required_ops[g->n_required++], WMIR_MAX_OP_NAME,
                         "%s", req->kids[i]->str);
        }
    }
    for (int i = 0; i < g->n_required; i++) {
        WmirOpKind k = wmir_op_parse(g->required_ops[i]);
        if (k == WMIR_OP_UNKNOWN || !wmir_kernel_supported(k)) {
            fprintf(stderr, "[wmir] missing kernel for required op '%s'\n",
                    g->required_ops[i]);
            free(buf); free(arena);
            memset(g, 0, sizeof(*g));
            return 0;
        }
    }

    jval *layers = json_get(wmir, "layers");
    if (layers && layers->t == J_ARR) {
        for (int i = 0; i < layers->len && g->n_layers < WMIR_MAX_LAYERS; i++) {
            if (!wmir_parse_layer(layers->kids[i], &g->layers[g->n_layers]))
                continue;
            g->n_layers++;
        }
    }
    if (g->desc.layers <= 0) g->desc.layers = g->n_layers;
    free(buf);
    free(arena);
    return g->n_layers > 0 && g->desc.hidden > 0;
}

/* Synthesize a dense GQA+SwiGLU/GELU WMIR layer graph from WhDesc (legacy packs). */
static void wmir_synthesize_from_desc(const WhDesc *d, WmirGraph *g) {
    memset(g, 0, sizeof(*g));
    g->present = 1;
    g->version = 1;
    g->text_only = 1;
    snprintf(g->family, sizeof(g->family), "%s", d->model_type);
    g->desc = *d;
    g->n_layers = d->layers;
    for (int i = 0; i < d->layers && i < WMIR_MAX_LAYERS; i++) {
        WmirLayer *L = &g->layers[i];
        L->n_ops = 0;
        L->kv_share_from = -1;
        /* attn */
        {
            WmirOp *op = &L->ops[L->n_ops++];
            memset(op, 0, sizeof(*op));
            op->kv_share_from = -1;
            op->kind = WMIR_OP_ATTN_GQA;
            snprintf(op->name, sizeof(op->name), "attn_gqa");
            if (d->sliding_window > 0) {
                int is_sw = d->sw_pattern > 0
                    ? ((i % d->sw_pattern) != (d->sw_pattern - 1)) : 1;
                if (is_sw) op->sliding_window = d->sliding_window;
            }
        }
        /* mlp */
        {
            WmirOp *op = &L->ops[L->n_ops++];
            memset(op, 0, sizeof(*op));
            op->kv_share_from = -1;
            if (d->act == WH_ACT_GELU_TANH) {
                op->kind = WMIR_OP_MLP_GELU;
                snprintf(op->name, sizeof(op->name), "mlp_gelu");
            } else {
                op->kind = WMIR_OP_MLP_SWIGLU;
                snprintf(op->name, sizeof(op->name), "mlp_swiglu");
            }
        }
        wmir_resolve_layer(L);
        if (d->sliding_window > 0)
            L->is_sw = d->sw_pattern > 0
                ? ((i % d->sw_pattern) != (d->sw_pattern - 1)) : 1;
    }
}

#endif /* WH_WMIR_H */
