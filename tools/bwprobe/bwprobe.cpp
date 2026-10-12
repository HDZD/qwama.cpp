// Fork-local bandwidth probe for the offload math in PLAN.md M0: the host RAM
// read ceiling and the Q4_K expert GEMV bandwidth (B_H). The PCIe side (B_P)
// lives in bwprobe-cuda.cu. Not intended for upstream.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static double now_s(void) {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static bool arg_int(const char ** args, int n, const char * name, int & out) {
    for (int i = 1; i + 1 < n; i++) {
        if (strcmp(args[i], name) == 0) {
            out = atoi(args[i + 1]);
            return true;
        }
    }
    return false;
}

// each thread streams its own slice so the working set stays DRAM sized;
// four accumulators keep the loop DRAM bound instead of ALU bound
static double measure_stream_read(int n_threads, double gib, int iters) {
    const size_t n = ((size_t) (gib * 1073741824.0 / sizeof(float))) & ~(size_t) 15;
    std::vector<float> buf(n, 1.0f);

    const size_t per = n / (size_t) n_threads;
    std::vector<double> acc((size_t) n_threads, 0.0);
    std::atomic<int> ready(0);
    std::atomic<bool> go(false);

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            const float * s = buf.data() + (size_t) t * per;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {}
            for (int it = 0; it < iters; it++) {
                for (size_t i = 0; i + 4 <= per; i += 4) {
                    a0 += s[i + 0];
                    a1 += s[i + 1];
                    a2 += s[i + 2];
                    a3 += s[i + 3];
                }
            }
            acc[(size_t) t] = a0 + a1 + a2 + a3;
        });
    }
    while (ready.load() < n_threads) {}
    const double t0 = now_s();
    go.store(true, std::memory_order_release);
    for (auto & th : threads) {
        th.join();
    }
    const double dt = now_s() - t0;

    double sink = 0.0;
    for (double a : acc) {
        sink += a;
    }
    if (sink < 0.0) {
        printf("");
    }
    return (double) n * sizeof(float) * (double) iters / dt / 1e9;
}

struct moe_dims {
    int hidden  = 2048;
    int inter   = 512;
    int experts = 256;
    int top_k   = 8;
};

// decode-shaped CPU expert GEMV: gate_up and down matmuls for top_k experts per
// step, expert ids rotating so every step reads a DRAM-sized working set
static double measure_host_moe(int n_threads, const moe_dims & d, int iters, int warmup) {
    const double t_start = now_s();
    const int64_t gu_rows      = 2 * d.inter;
    const size_t  gu_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, d.hidden);
    const size_t  d_row_bytes  = ggml_row_size(GGML_TYPE_Q4_K, d.inter);
    const size_t  gu_bytes     = gu_row_bytes * gu_rows;   // per expert
    const size_t  d_bytes      = d_row_bytes  * d.hidden;  // per expert
    const size_t  expert_bytes = gu_bytes + d_bytes;

    printf("host-moe: H=%d I=%d E=%d top_k=%d threads=%d expert_bytes=%zu\n",
           d.hidden, d.inter, d.experts, d.top_k, n_threads, expert_bytes);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        fprintf(stderr, "bwprobe: cpu backend init failed\n");
        exit(1);
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    fprintf(stderr, "[bwprobe] backend init: %.2fs\n", now_s() - t_start);

    struct ggml_init_params wip = {
        /* .mem_size   = */ ggml_tensor_overhead() * (2 * (size_t) d.experts + 16) +
                             (gu_bytes + d_bytes) * (size_t) d.experts + (16ull << 20),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * wctx = ggml_init(wip);

    struct ggml_tensor * gu_base = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, d.hidden, gu_rows, d.experts);
    struct ggml_tensor * d_base  = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, d.inter,  d.hidden, d.experts);
    fprintf(stderr, "[bwprobe] weight tensors: %.2fs\n", now_s() - t_start);

    {
        // one quantized expert set is enough: weight values do not affect the
        // bytes read per step, and k-quant search per block costs seconds per MB
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
        std::vector<float> src(gu_rows * d.hidden);
        for (auto & v : src) {
            v = dist(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), gu_base->data, 0, gu_rows, d.hidden, nullptr);
        src = std::vector<float>((size_t) d.inter * d.hidden);
        for (auto & v : src) {
            v = dist(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), d_base->data, 0, d.hidden, d.inter, nullptr);

        for (int e = 1; e < d.experts; e++) {
            memcpy((char *) gu_base->data + gu_bytes * (size_t) e, gu_base->data, gu_bytes);
            memcpy((char *) d_base->data  + d_bytes  * (size_t) e, d_base->data,  d_bytes);
        }
    }
    fprintf(stderr, "[bwprobe] weights fill: %.2fs\n", now_s() - t_start);

    std::vector<struct ggml_tensor *> gu(d.experts);
    std::vector<struct ggml_tensor *> dn(d.experts);
    for (int e = 0; e < d.experts; e++) {
        gu[e] = ggml_view_2d(wctx, gu_base, d.hidden, gu_rows, gu_row_bytes, gu_bytes * (size_t) e);
        dn[e] = ggml_view_2d(wctx, d_base,  d.inter,  d.hidden, d_row_bytes, d_bytes * (size_t) e);
    }
    fprintf(stderr, "[bwprobe] views: %.2fs\n", now_s() - t_start);

    std::vector<double> samples;
    for (int step = 0; step < warmup + iters; step++) {
        struct ggml_init_params gip = {
            /* .mem_size   = */ 64ull << 20,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        struct ggml_context * gctx = ggml_init(gip);

        struct ggml_tensor * x   = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, d.hidden);
        struct ggml_tensor * out = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, d.hidden);
        memset(out->data, 0, ggml_nbytes(out));
        {
            std::mt19937 rng((unsigned) step);
            std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
            float * xd = (float *) x->data;
            for (int i = 0; i < d.hidden; i++) {
                xd[i] = dist(rng);
            }
        }

        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        for (int k = 0; k < d.top_k; k++) {
            const int e = (step * d.top_k + k) % d.experts;
            struct ggml_tensor * y1 = ggml_mul_mat(gctx, gu[e], x);
            struct ggml_tensor * g  = ggml_view_1d(gctx, y1, d.inter, 0);
            struct ggml_tensor * u  = ggml_view_1d(gctx, y1, d.inter, (size_t) d.inter * sizeof(float));
            struct ggml_tensor * h  = ggml_mul(gctx, ggml_silu(gctx, g), u);
            struct ggml_tensor * y2 = ggml_mul_mat(gctx, dn[e], h);
            out = ggml_add(gctx, out, y2);
        }
        ggml_build_forward_expand(gf, out);

        const double t0 = now_s();
        const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
        const double dt = now_s() - t0;
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "bwprobe: graph compute failed\n");
            exit(1);
        }
        if (step < 3) {
            fprintf(stderr, "[bwprobe] step %d: %.3f ms (graph %.2f MB)\n", step, dt * 1e3, (double) ggml_graph_size(gf));
        }
        if (step >= warmup) {
            samples.push_back(dt);
        }
        ggml_free(gctx);
    }
    fprintf(stderr, "[bwprobe] loop total: %.2fs\n", now_s() - t_start);

    ggml_free(wctx);
    ggml_backend_free(backend);

    std::sort(samples.begin(), samples.end());
    const double med = samples[samples.size() / 2];
    return (double) d.top_k * (double) expert_bytes / med / 1e9;
}

int main(int argc, const char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const char * mode = "all";
    if (argc > 1 && argv[1][0] != '-') {
        mode = argv[1];
    }

    int threads   = 14;
    int gib       = 2;
    int iters     = 6;
    int hm_iters  = 64;
    int warmup    = 8;
    moe_dims dims;

    arg_int(argv, argc, "--threads", threads);
    arg_int(argv, argc, "--gib",     gib);
    arg_int(argv, argc, "--iters",   iters);
    arg_int(argv, argc, "--warmup",  warmup);
    arg_int(argv, argc, "--hidden",  dims.hidden);
    arg_int(argv, argc, "--inter",   dims.inter);
    arg_int(argv, argc, "--experts", dims.experts);
    arg_int(argv, argc, "--topk",    dims.top_k);
    arg_int(argv, argc, "--hm-iters", hm_iters);

    const bool do_stream = strcmp(mode, "stream")   == 0 || strcmp(mode, "all") == 0;
    const bool do_moe    = strcmp(mode, "host-moe") == 0 || strcmp(mode, "all") == 0;

    if (!do_stream && !do_moe) {
        fprintf(stderr, "usage: bwprobe [stream|host-moe|all] [--threads N] [--gib N] [--iters N]\n");
        return 1;
    }

    if (do_stream) {
        const double gbs = measure_stream_read(threads, (double) gib, iters);
        printf("stream_gbs=%.2f\n", gbs);
        printf("stream: threads=%d buffer=%d GiB iters=%d\n", threads, gib, iters);
    }
    if (do_moe) {
        const double gbs = measure_host_moe(threads, dims, hm_iters, warmup);
        printf("b_h_gbs=%.2f\n", gbs);
    }
    return 0;
}
