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

#define Mb(x) (x * 1024 * 1024)

const uint32_t read_size = Mb(12);
const uint32_t alloc_size = Mb(128);

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
        /* .mem_size = */ alloc_size,
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * tensor = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, alloc_size);
    ggml_format_name(tensor, "tensor_test");
    ggml_graph_add_node(gf, tensor);
    ggml_build_forward_expand(gf, tensor);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != NULL);
    
    std::string cmd = "dd if=/dev/zero of=test.bin bs=1M count=" + std::to_string(alloc_size / Mb(1));
    int ret = system(cmd.c_str());
    GGML_ASSERT(ret == 0);

    llama_file file("test.bin", "rb");
    for (uint32_t i = 0; i < 10; i++) {
        uint32_t offset = i * read_size;

        ggml_backend_tensor_set_device(tensor, file.file_id(), read_size, offset, offset);
        void *temp_buf = malloc(read_size);
        file.read_raw(temp_buf, read_size);

        void *temp_buf1 = malloc(read_size);
        ggml_backend_tensor_get(tensor, temp_buf, offset, read_size);
        
        GGML_ASSERT(memcmp(temp_buf, temp_buf1, read_size) == 0);

        free(temp_buf);
        free(temp_buf1);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::remove("test.bin");

    return 0;
}
