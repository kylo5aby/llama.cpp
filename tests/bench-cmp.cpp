#include <cassert>
#include <cstddef>
#include <cstdio>
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include "../src/llama-mmap.h"

#include <thread>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <time.h>

#define MB(x) (x * 1024 * 1024)
#define KB(x) (x * 1024)
#define GB(x) (x * 1024L * 1024L * 1024L)

const uint32_t kb_size = KB(4);
const uint32_t mb_size = MB(4);
const uint64_t limit_size = 1024L * (mb_size + kb_size); // ~4GB
const uint64_t gpu_alloc_size = GB(1);
const char *path = "/root/zhenwei/llama.cpp/build-bench/bin/test_5g.bin";
std::vector<uint8_t> temp_buf1(kb_size);
std::vector<uint8_t> temp_buf2(mb_size + kb_size);


int64_t time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

void bench_fixed_offset(ggml_tensor * tensor, const llama_file &file) {
    uint64_t total_size = 0;
    size_t file_offset = 0;
    size_t buf_offset = 0;

    while (total_size < limit_size) {
        ggml_backend_tensor_set_device(tensor, file.file_id(), (mb_size + kb_size), file_offset, buf_offset);
        total_size += (mb_size + kb_size);
    }

    printf("(GDS: bench_fixed_offset) Total size: %zu MB\n", total_size / MB(1));
}

void bench_ordered_offset(ggml_tensor * tensor, const llama_file &file) {
    uint64_t total_size = 0;
    size_t file_offset = 0;
    size_t buf_offset = 0;

    while (total_size < limit_size) {
        ggml_backend_tensor_set_device(tensor, file.file_id(), (mb_size + kb_size), file_offset, buf_offset);
        file_offset += (mb_size + kb_size);
        buf_offset += (mb_size + kb_size);
        if (buf_offset >= gpu_alloc_size) {
            buf_offset = 0;
        }
        total_size += (mb_size + kb_size);
    }
    file.seek(0, SEEK_SET);

    printf("(GDS: bench_ordered_offset) Total size: %zu MB\n", total_size / MB(1));
}

void bench_buf_4k_gds_4m(ggml_tensor * tensor, const llama_file &file) {
    uint64_t total_size = 0;
    size_t file_offset = 0;
    size_t buf_offset = 0;

    while (total_size < limit_size) {
        file.read_raw(temp_buf1.data(), kb_size);
        file_offset += kb_size;
        ggml_backend_tensor_set_device(tensor, file.file_id(), mb_size, file_offset, buf_offset);
        file_offset += mb_size;
        buf_offset += mb_size;
        if (buf_offset >= gpu_alloc_size) {
            buf_offset = 0;
        }
        total_size += (mb_size + kb_size);
    }
    file.seek(0, SEEK_SET);

    printf("(GDS: bench_buf_4k_gds_4m) Total size: %zu MB\n", total_size / (MB(1)));
}

void bench_buffer(ggml_tensor * tensor, const llama_file &file) {
    uint64_t total_size = 0;
    size_t buf_offset = 0;

    while (total_size < limit_size) {
        file.read_raw(temp_buf2.data(), mb_size + kb_size);
        ggml_backend_tensor_set(tensor, temp_buf2.data(), buf_offset, mb_size + kb_size);
        buf_offset += mb_size + kb_size;
        if (buf_offset >= gpu_alloc_size) {
            buf_offset = 0;
        }
        total_size += (mb_size + kb_size);
    }
    file.seek(0, SEEK_SET);

    printf("(GDS: bench_buffer) Total size: %zu MB\n", total_size / MB(1));
}

int main() {    
    ggml_backend_load_all();
    ggml_backend_dev_t dev;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        dev = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(dev), "SYCL", 4) == 0) {
            break;
        }
    }

    if (strncmp(ggml_backend_dev_name(dev), "SYCL", 4) != 0) {
        printf("SYCL backend not found\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    GGML_ASSERT(backend != NULL);

    ggml_init_params params = {
        /* .mem_size = */ gpu_alloc_size,
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * tensor = ::ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, gpu_alloc_size);
    ggml_format_name(tensor, "tensor_test");
    ggml_graph_add_node(gf, tensor);
    ggml_build_forward_expand(gf, tensor);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != NULL);

    llama_file file(path, "rb");

    int64_t time_start = 0;
    int64_t time_end = 0;

    printf("(GDS: bench_fixed_offset) Start\n");
    time_start = time_ms();
    bench_fixed_offset(tensor, file);
    time_end = time_ms();
    printf("(GDS: bench_fixed_offset) Time: %ld ms\n", time_end - time_start);

    printf("\n ------------------------------------ \n (GDS: bench_ordered_offset) Start\n");
    time_start = time_ms();
    bench_ordered_offset(tensor, file);
    time_end = time_ms();
    printf("(GDS: bench_ordered_offset) Time: %ld ms\n", time_end - time_start);

    printf("\n ------------------------------------ \n (GDS: bench_buf_4k_gds_4m) Start\n");
    time_start = time_ms();
    bench_buf_4k_gds_4m(tensor, file);
    time_end = time_ms();
    printf("(GDS: bench_buf_4k_gds_4m) Time: %ld ms\n", time_end - time_start);

    printf("\n ------------------------------------ \n (GDS: bench_buffer) Start\n");
    time_start = time_ms();
    bench_buffer(tensor, file);
    time_end = time_ms();
    printf("(GDS: bench_buffer) Time: %ld ms\n", time_end - time_start);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
