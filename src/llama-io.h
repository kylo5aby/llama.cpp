#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;

class llama_io_write_i {
public:
    llama_io_write_i() = default;
    virtual ~llama_io_write_i() = default;

    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) = 0;

    // bytes written so far
    virtual size_t n_bytes() = 0;

    void write_string(const std::string & str);

    // for debug only
    virtual size_t tell() {
        return 0;
    }
};

class llama_io_read_i {
public:
    llama_io_read_i() = default;
    virtual ~llama_io_read_i() = default;

    virtual const uint8_t * read(size_t size) = 0;
    virtual void read_to(void * dst, size_t size) = 0;

    // bytes read so far
    virtual size_t n_bytes() = 0;

    // for debug only
    virtual void update_size_read(size_t size) {
        // do nothing
    }

    // for debug only
    virtual int get_fd() {
        return -1;
    }

    // for debug only
    virtual void seek(size_t offset, int whence) const {
        // do nothing
    }

    // for debug only
    virtual size_t tell() const {
        return 0;
    }

    void read_string(std::string & str);
};
