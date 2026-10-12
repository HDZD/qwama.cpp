// PCIe bandwidth probe (B_P): pinned linear H2D/D2H copies plus an
// expert-sized batched gather, shaped like a decode miss upload.
// Pure cudart - no cuBLAS, so it is safe to build with any CUDA 13.x and
// says nothing about the CUDA 13.2 cuBLAS issue (IDEA.md C1).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#define CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "bwprobe-cuda: %s failed: %s\n", #x, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static bool arg_int(const char ** args, int n, const char * name, int & out) {
    for (int i = 1; i + 1 < n; i++) {
        if (strcmp(args[i], name) == 0) {
            out = atoi(args[i + 1]);
            return true;
        }
    }
    return false;
}

int main(int argc, const char ** argv) {
    int mib          = 256;
    int iters        = 30;
    int experts      = 64;
    int expert_kib   = 1728;  // qwen3.6-35b-a3b Q4_K expert: 1.6875 MiB
    int gather_iters = 20;

    arg_int(argv, argc, "--mib",          mib);
    arg_int(argv, argc, "--iters",        iters);
    arg_int(argv, argc, "--experts",      experts);
    arg_int(argv, argc, "--expert-kib",   expert_kib);
    arg_int(argv, argc, "--gather-iters", gather_iters);

    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s (pci %04x:%02x:%02x)\n", prop.name,
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    // linear pinned H2D / D2H
    const size_t bytes = (size_t) mib << 20;
    void * h = nullptr;
    void * d = nullptr;
    CHECK(cudaHostAlloc(&h, bytes, cudaHostAllocDefault));
    CHECK(cudaMalloc(&d, bytes));
    memset(h, 0x5a, bytes);

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    float h2d = 0.0f, d2h = 0.0f;
    for (int w = 0; w < 3; w++) {
        CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0));
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(ev0, 0));
    for (int i = 0; i < iters; i++) {
        CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0));
    }
    CHECK(cudaEventRecord(ev1, 0));
    CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    h2d = (float) ((double) bytes * iters / (ms / 1e3) / 1e9);

    CHECK(cudaEventRecord(ev0, 0));
    for (int i = 0; i < iters; i++) {
        CHECK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, 0));
    }
    CHECK(cudaEventRecord(ev1, 0));
    CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    d2h = (float) ((double) bytes * iters / (ms / 1e3) / 1e9);

    printf("pcie_h2d_gbs=%.2f\n", h2d);
    printf("pcie_d2h_gbs=%.2f\n", d2h);

    // expert-sized gather: many small H2D copies from scattered pinned offsets,
    // matching the shape of one decode miss (or one full layer upload)
    const size_t eb     = (size_t) expert_kib << 10;
    const size_t pool   = std::max<size_t>(64ull << 20, eb * (size_t) experts * 4);
    const size_t maxoff = pool - eb;

    void * hp = nullptr;
    void * dp = nullptr;
    CHECK(cudaHostAlloc(&hp, pool, cudaHostAllocDefault));
    CHECK(cudaMalloc(&dp, eb * (size_t) experts));
    memset(hp, 0x3c, pool);

    std::mt19937 rng(7);
    std::uniform_int_distribution<size_t> dist(0, maxoff / 4096 * 4096);
    std::vector<size_t> offsets((size_t) experts);
    for (auto & o : offsets) {
        o = dist(rng);
    }

    for (int w = 0; w < 2; w++) {
        for (int e = 0; e < experts; e++) {
            CHECK(cudaMemcpyAsync((char *) dp + eb * (size_t) e, (char *) hp + offsets[(size_t) e],
                                  eb, cudaMemcpyHostToDevice, 0));
        }
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaEventRecord(ev0, 0));
    for (int i = 0; i < gather_iters; i++) {
        for (int e = 0; e < experts; e++) {
            CHECK(cudaMemcpyAsync((char *) dp + eb * (size_t) e, (char *) hp + offsets[(size_t) e],
                                  eb, cudaMemcpyHostToDevice, 0));
        }
    }
    CHECK(cudaEventRecord(ev1, 0));
    CHECK(cudaEventSynchronize(ev1));
    CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const double gather = (double) eb * experts * gather_iters / (ms / 1e3) / 1e9;
    printf("pcie_gather_gbs=%.2f\n", gather);
    printf("gather: %d experts x %d KiB x %d iters\n", experts, expert_kib, gather_iters);

    CHECK(cudaFree(dp));
    CHECK(cudaFreeHost(hp));
    CHECK(cudaFree(d));
    CHECK(cudaFreeHost(h));
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    return 0;
}
