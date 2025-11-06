#include <cassert>
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

#define Mb(x) (x * 1024 * 1024)

uint32_t read_size = Mb(32);
const uint32_t alloc_size = Mb(1024);
bool verify_success = true;

int64_t time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}


bool parse_args(int argc, char *argv[], std::string &type, bool &verify, uint32_t &offset) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-size") == 0 && i + 1 < argc) {
            read_size = Mb(atoi(argv[i + 1]));
            i++;
        } else if (strcmp(argv[i], "-type") == 0 && i + 1 < argc) {
            type = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verify") == 0) {
            verify = true;
            i++;
        } else if (strcmp(argv[i], "-offset") == 0 && i + 1 < argc) {
            offset = atoi(argv[i + 1]);
            i++;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-size MB] [-type buffer|gds] [-v]\n", argv[0]);
            printf("  -size: read size in MB (default: 32)\n");
            printf("  -type: read type - buffer or gds (default: buffer)\n");
            printf("  -v, --verify: enable verification\n");
            printf("  -offset: offset in bytes (default: 0)\n");
            exit(0);
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    std::string read_type = "buffer";
    bool verify = false;
    uint32_t offset = 0;
    parse_args(argc, argv, read_type, verify, offset);

    printf("Configuration:\n");
    printf("  Read size: %u MB\n", read_size / Mb(1));
    printf("  Read type: %s\n", read_type.c_str());
    printf("  Verification: %s\n", verify ? "enabled" : "disabled");
    printf("\n");

    // std::string cmd = "dd if=/dev/zero of=test.bin bs=1M count=" + std::to_string(alloc_size / Mb(1));
    // int ret = system(cmd.c_str());
    // GGML_ASSERT(ret == 0);
    
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
        /* .mem_size = */ alloc_size,
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * tensor = ::ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, alloc_size);
    ggml_format_name(tensor, "tensor_test");
    ggml_graph_add_node(gf, tensor);
    ggml_build_forward_expand(gf, tensor);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != NULL);

    llama_file file("test_5g.bin", "rb");

    const uint32_t n_iter = 32;
    const uint32_t e = alloc_size / read_size;
    std::vector<uint8_t> temp_buf(read_size);
    assert(file.file_id() != -1);
    uint64_t total_read_size = 0;
    printf("Start measuring time\n");
    int64_t time_start = time_ms();
    for (uint32_t i = 0; i < n_iter; i++) {
        for (uint32_t j = 0; j < e; j++) {
            uint32_t offset = j * read_size;
            
            uint32_t first_read_start = 0;
            if (i == 0 && j == 0) {
                first_read_start = time_ms();
            }
            if (read_type == "gds") {
                // Use GDS direct read
                ggml_backend_tensor_set_device(tensor, file.file_id(), read_size, offset, offset);
                
                if (verify) {
                    // Verification: get data back and compare
                    ggml_backend_tensor_get(tensor, temp_buf.data(), offset, read_size);
                    std::vector<uint8_t> temp_buf1(read_size);
                    file.read_raw(temp_buf1.data(), read_size);
                    
                    // Compare the two buffers
                    bool match = memcmp(temp_buf.data(), temp_buf1.data(), read_size) == 0;
                    if (!match) {
                        printf("Verification failed at offset %u\n", offset);
                        verify_success = false;
                        return 1;
                    }
                }
            } else {
                // Use buffer read (default)
                file.read_raw(temp_buf.data(), read_size);
                ggml_backend_tensor_set(tensor, temp_buf.data(), offset, read_size);
            }

            if (i == 0 && j == 0) {
                int64_t first_read_end = time_ms();
                printf("First read time: %ld ms\n", first_read_end - first_read_start);
            }
            total_read_size += read_size;
        }
        file.seek(0, SEEK_SET);
    }

    int64_t time_end = time_ms();
    printf("time: %ld ms\n", time_end - time_start);

    printf("Total read size: %zu MB\n", total_read_size / 1024 / 1024);
    if (verify) {
        printf("Verification: %s\n", verify_success ? "PASSED" : "FAILED");
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return 0;
}
