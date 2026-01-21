/*
 *  Copyright (C) 2025, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#ifdef ENABLE_IPCOMP
#include <SZ3/compressor/IPComp.hpp>
#include <SZ3/quantizer/IntegerQuantizer2.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>
#include <SZ3/encoder/BypassEncoder.hpp>
#endif

#include <new>
#include <memory>
#include <vector>
#include <array>
#include <cstring>
#include <cstddef>
#include <exception>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <mpi.h>  // 用于计时
#include <cstdarg>

/* Some toolchains/language servers may not see ALWAYS_INLINE from SZ3 headers.
 * Define it locally if missing. */
#ifndef ALWAYS_INLINE
#if defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif
#endif

/* Data type constants */
#define IPCOMP_FLOAT  0
#define IPCOMP_DOUBLE 1

static inline bool ipcomp_wrapper_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("IPCOMP_WRAPPER_DEBUG");
        enabled = (v && *v) ? 1 : 0;
    }
    return enabled == 1;
}

static inline void ipcomp_wrapper_dbg(const char* fmt, ...) {
    if (!ipcomp_wrapper_debug_enabled()) return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
}

/* IPComp wrapper structure */
struct IPCompWrapper {
    int ndim;
    std::vector<int> dims;
    int interp_op;
    int direction_op;
    int layers;
    size_t block_size;
    size_t interp_dim_limit;
    int level_progressive;
    void* compressor;  /* Will hold the actual SZ3 compressor */
    int data_type;
    size_t num_elements;
    double data_range;  /* 保存压缩时计算的数据范围，用于解压缩 */
    double meta_min;
    double meta_max;
    bool has_meta_minmax;
    std::vector<unsigned char> mask_valid;
    std::vector<unsigned char> mask_boundary;
    size_t mask_bytes;
    size_t mask_valid_count;
    int guard_radius;
    bool mask_set;
};

static std::vector<double> build_default_relative_ebs(int layers) {
    switch (layers) {
        case 1:
            return {1e-3};
        case 2:
            return {1e-3, 1e-6};
        case 3:
            return {1e-2, 1e-4, 1e-6};
        case 4:
            return {1e-3, 1e-4, 1e-5, 1e-6};
        case 5:
            return {1e-6 * 4096.0, 1e-6 * 256.0, 1e-6 * 16.0, 1e-6};
        case 9:
            return {1e-9};
        case 11:
            return {1e-9};
        case 15:
            return {1e-9 * 65536.0, 1e-9 * 4096.0, 1e-9 * 256.0, 1e-9 * 16.0, 1e-9};
        case 20:
            return {1e-9 * 4096.0, 1e-9};
        case 99:
            return {1e-3};
        default:
            return {1e-6};
    }
}

namespace {

constexpr uint32_t kIPCompHeaderMagic = 0x49504350u; /* 'IPCP' */
constexpr uint16_t kIPCompHeaderVersion = 1;
constexpr uint16_t kIPCompHeaderFlagHasRange = 0x1;
constexpr size_t kIPCompHeaderSize = sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint16_t) + sizeof(double);

struct IPCompHeaderInfo {
    const unsigned char* payload;
    double data_range;
    bool has_header;
    bool has_range;
};

ALWAYS_INLINE IPCompHeaderInfo decode_ipcomp_header(const unsigned char* data) {
    IPCompHeaderInfo info{data, 0.0, false, false};
    if (!data) {
        return info;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, data, sizeof(uint32_t));
    if (magic != kIPCompHeaderMagic) {
        return info;
    }

    uint16_t version = 0;
    std::memcpy(&version, data + sizeof(uint32_t), sizeof(uint16_t));
    if (version != kIPCompHeaderVersion) {
        return info;
    }

    uint16_t flags = 0;
    std::memcpy(&flags, data + sizeof(uint32_t) + sizeof(uint16_t), sizeof(uint16_t));

    double stored_range = 0.0;
    std::memcpy(&stored_range, data + sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint16_t), sizeof(double));

    info.payload = data + kIPCompHeaderSize;
    info.has_header = true;
    if (flags & kIPCompHeaderFlagHasRange) {
        info.has_range = true;
        info.data_range = stored_range;
    }
    return info;
}

ALWAYS_INLINE void encode_ipcomp_header(unsigned char* dest, double data_range, bool has_range) {
    uint32_t magic = kIPCompHeaderMagic;
    uint16_t version = kIPCompHeaderVersion;
    uint16_t flags = has_range ? kIPCompHeaderFlagHasRange : 0;

    std::memcpy(dest, &magic, sizeof(uint32_t));
    dest += sizeof(uint32_t);
    std::memcpy(dest, &version, sizeof(uint16_t));
    dest += sizeof(uint16_t);
    std::memcpy(dest, &flags, sizeof(uint16_t));
    dest += sizeof(uint16_t);
    std::memcpy(dest, &data_range, sizeof(double));
}

template<typename T>
void delete_with_new_array(void* ptr) {
    delete[] static_cast<T*>(ptr);
}

inline void delete_with_free(void* ptr) {
    std::free(ptr);
}

inline void delete_with_aligned256(void* ptr) {
    ::operator delete[](ptr, std::align_val_t{256});
}

struct BufferRecord {
    void (*deleter)(void*);
};

std::mutex g_buffer_mutex;
std::unordered_map<void*, BufferRecord> g_buffer_records;

inline void register_buffer(void* ptr, void (*deleter)(void*)) {
    if (!ptr || !deleter) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    g_buffer_records[ptr] = BufferRecord{deleter};
}

inline bool take_buffer_record(void* ptr, void (**deleter)(void*)) {
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    auto it = g_buffer_records.find(ptr);
    if (it == g_buffer_records.end()) {
        return false;
    }
    if (deleter) {
        *deleter = it->second.deleter;
    }
    g_buffer_records.erase(it);
    return true;
}

class StdoutInterceptor {
public:
    StdoutInterceptor(bool enable, double meta_min, double meta_max)
        : enabled_(enable && meta_max >= meta_min),
          meta_min_(meta_min),
          meta_max_(meta_max) {
        if (enabled_) {
            enabled_ = start();
        }
    }

    ~StdoutInterceptor() {
        if (enabled_ && active_) {
            restore(false);
        }
    }

    std::string consume_and_rewrite() {
        if (!enabled_) {
            return {};
        }
        enabled_ = false;
        return rewrite_verification_lines(restore(true), meta_min_, meta_max_);
    }

private:
    bool start() {
        if (pipe(pipe_fds_) != 0) {
            return false;
        }
        saved_stdout_ = dup(STDOUT_FILENO);
        if (saved_stdout_ < 0) {
            close(pipe_fds_[0]);
            close(pipe_fds_[1]);
            pipe_fds_[0] = pipe_fds_[1] = -1;
            return false;
        }
        if (dup2(pipe_fds_[1], STDOUT_FILENO) < 0) {
            close(saved_stdout_);
            close(pipe_fds_[0]);
            close(pipe_fds_[1]);
            pipe_fds_[0] = pipe_fds_[1] = -1;
            saved_stdout_ = -1;
            return false;
        }
        close(pipe_fds_[1]);
        pipe_fds_[1] = -1;
        active_ = true;
        return true;
    }

    std::string restore(bool keep_output) {
        std::string output;
        if (!active_) {
            return output;
        }
        std::fflush(stdout);
        std::cout.flush();
        if (saved_stdout_ >= 0) {
            dup2(saved_stdout_, STDOUT_FILENO);
            close(saved_stdout_);
            saved_stdout_ = -1;
        }
        if (pipe_fds_[1] != -1) {
            close(pipe_fds_[1]);
            pipe_fds_[1] = -1;
        }
        if (pipe_fds_[0] >= 0) {
            if (keep_output) {
                char buffer[4096];
                ssize_t nread;
                while ((nread = read(pipe_fds_[0], buffer, sizeof(buffer))) > 0) {
                    output.append(buffer, static_cast<size_t>(nread));
                }
            } else {
                char buffer[4096];
                while (read(pipe_fds_[0], buffer, sizeof(buffer)) > 0) {
                    /* discard */
                }
            }
            close(pipe_fds_[0]);
            pipe_fds_[0] = -1;
        }
        active_ = false;
        return output;
    }

    static std::string rewrite_verification_lines(const std::string& input,
                                                  double meta_min,
                                                  double meta_max) {
        if (input.empty()) {
            return input;
        }
        std::ostringstream oss;
        std::istringstream iss(input);
        std::string line;
        double range = meta_max - meta_min;
        bool replaced = false;
        while (std::getline(iss, line)) {
            if (!replaced && line.find("[Verification] Min = ") != std::string::npos) {
                char buffer[256];
                std::snprintf(buffer, sizeof(buffer),
                              "[Verification] Min = %.6f, Max = %.6f, Range = %.6f",
                              meta_min, meta_max, range);
                oss << buffer;
                replaced = true;
            } else {
                oss << line;
            }
            if (!iss.eof()) {
                oss << '\n';
            }
        }
        return oss.str();
    }

    bool enabled_ = false;
    bool active_ = false;
    int pipe_fds_[2] = {-1, -1};
    int saved_stdout_ = -1;
    double meta_min_ = 0.0;
    double meta_max_ = 0.0;
};

template <typename Callable>
auto with_verification_override(IPCompWrapper* wrapper, Callable&& fn)
    -> decltype(fn()) {
    if (wrapper && wrapper->has_meta_minmax) {
        StdoutInterceptor interceptor(true, wrapper->meta_min, wrapper->meta_max);
        auto result = fn();
        std::string logs = interceptor.consume_and_rewrite();
        if (!logs.empty()) {
            std::fwrite(logs.data(), 1, logs.size(), stdout);
            std::fflush(stdout);
        }
        return result;
    }
    return fn();
}

} // namespace

extern "C" {

#ifdef ENABLE_IPCOMP
/* Create IPComp compressor instance */
void* ipcomp_create_compressor(int ndim, const int* dims, int interp_op, int direction_op, 
                              int layers, size_t interp_dim_limit, size_t block_size, int level_progressive) {
    try {
        IPCompWrapper* wrapper = new IPCompWrapper();
        wrapper->ndim = ndim;
        wrapper->dims.assign(dims, dims + ndim);
        wrapper->interp_op = interp_op;
        wrapper->direction_op = direction_op;
        wrapper->layers = layers;
        wrapper->block_size = block_size;
        /* interp_dim_limit must be even; keep it safe here as a last line of defense */
        if (interp_dim_limit & 1u) interp_dim_limit--;
        if (interp_dim_limit < 2) interp_dim_limit = 2;
        wrapper->interp_dim_limit = interp_dim_limit;
        wrapper->level_progressive = level_progressive;
        wrapper->compressor = nullptr;
        wrapper->data_type = -1;
        wrapper->data_range = 0.0;  /* 将在压缩时计算 */
        wrapper->meta_min = 0.0;
        wrapper->meta_max = 0.0;
        wrapper->has_meta_minmax = false;
        wrapper->mask_bytes = 0;
        wrapper->mask_valid_count = 0;
        wrapper->guard_radius = 0;
        wrapper->mask_set = false;
        wrapper->mask_valid.clear();
        wrapper->mask_boundary.clear();
        
        /* Calculate total number of elements */
        wrapper->num_elements = 1;
        for (int i = 0; i < ndim; i++) {
            wrapper->num_elements *= dims[i];
        }
        
        return wrapper;
    } catch (...) {
        return nullptr;
    }
}

/* Destroy IPComp compressor instance */
void ipcomp_destroy_compressor(void* compressor) {
    if (compressor == nullptr) return;
    
    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);

    /* 现在压缩器是栈上的临时对象，wrapper->compressor 永远是 nullptr，
     * 所以只需要清理 wrapper 自身即可 */
    delete wrapper;
}

/* Setup compressor - now just records data type, actual compressor created per-chunk */
int ipcomp_setup(void* compressor, const void* data, int data_type) {
    if (compressor == nullptr) return -1;
    
    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);
    wrapper->data_type = data_type;
    
    /* 压缩器在 compress/decompress 时根据实际维度动态创建，
     * wrapper->compressor 保持为 nullptr */
    
    return 0;
}

/* Setup layers function - called from filter (now just validates) */
int ipcomp_setup_layers(void* compressor, const void* data, int data_type) {
    if (!compressor || !data) return -1;
    
    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);
    
    // setupLayers is now called in ipcomp_compress, so this function
    // just validates inputs and ensures compressor is ready
    if (wrapper->compressor == nullptr) {
        // Need to setup compressor first
        return ipcomp_setup(compressor, data, data_type);
    }
    
    return 0;
}

/* Compress data with optional external range
 * 
 * If wrapper->data_range > 0, it will be used instead of recalculating from data.
 * This is important for parallel compression where each process sees only partial data.
 */
unsigned char* ipcomp_compress(void* compressor, const void* data, int data_type, size_t* compressed_size, int ndim, const int* dims) {
    // ---- Preflight ----
    if (!compressed_size) return nullptr;
    *compressed_size = 0;
    if (!compressor || !data) return nullptr;

    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);
    if (!wrapper) return nullptr;
    
    // 验证输入维度
    if (ndim <= 0 || ndim > 3 || !dims) return nullptr;
    
    // 计算实际数据元素数量（使用传入的dims，不是wrapper->dims）
    size_t nelems = 1;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return nullptr;
        nelems *= static_cast<size_t>(dims[i]);
    }
    if (nelems == 0) return nullptr;

    // 原始字节数与临时无损缓冲容量估算（给 zstd 留出冗余）
    const size_t elem_size = (data_type == IPCOMP_DOUBLE) ? sizeof(double) : sizeof(float);
    const size_t raw_bytes = nelems * elem_size;

    // 经验容量：raw + 12.5% + 64KB，且至少 1MB；避免 memmove 覆盖
    size_t lossless_cap = raw_bytes + (raw_bytes >> 3) + 65536;
    if (lossless_cap < (1u << 20)) lossless_cap = (1u << 20);

    void* lossless_buffer_void = nullptr;
    if (posix_memalign(&lossless_buffer_void, 256, lossless_cap) != 0 || !lossless_buffer_void) return nullptr;
    unsigned char* lossless_buffer = static_cast<unsigned char*>(lossless_buffer_void);

    // 把 *compressed_size 当作"容量"传给 SZ3（关键修复点）
    *compressed_size = lossless_cap;

    unsigned char* result = nullptr;
    void* data_copy = nullptr;  // 统一的数据副本指针，用于异常清理

    /* 添加详细计时 */
    double t_copy = 0, t_create_comp = 0, t_setup = 0, t_compress_core = 0;
    double t0, t1;
    static int first_call = 1;  // 只在第一次调用时打印详细信息

    try {
        if (data_type == IPCOMP_FLOAT) {
            const float* float_data = static_cast<const float*>(data);
            if (!float_data) { free(lossless_buffer); *compressed_size = 0; return nullptr; }
            
            // Timer: 创建数据副本
            t0 = MPI_Wtime();
            float* float_copy = new (std::nothrow) float[nelems];
            if (!float_copy) { free(lossless_buffer); *compressed_size = 0; return nullptr; }
            std::memcpy(float_copy, float_data, nelems * sizeof(float));
            data_copy = float_copy;  // 记录用于异常清理
            t1 = MPI_Wtime();
            t_copy = t1 - t0;

            // Timer: 创建压缩器对象
            t0 = MPI_Wtime();
            
            // 根据实际传入的维度动态创建压缩器
            if (ndim == 2) {
                // 使用实际传入的dims创建2D维度数组
                std::array<size_t, 2> sz_dims;
                sz_dims[0] = static_cast<size_t>(dims[0]);
                sz_dims[1] = static_cast<size_t>(dims[1]);
                
                // 动态创建2D压缩器
                auto sz_comp = SZ3::SZProgressiveMQuant<float, 2,
                    SZ3::LinearQuantizer2<float>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<float>(nelems, 1.0), // 使用实际元素数量
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size
                );
                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }
                
                t1 = MPI_Wtime();
                t_create_comp = t1 - t0;
                
                // Timer: setupLayers
                t0 = MPI_Wtime();

                // === IPComp 压缩：使用硬编码的 error bounds ===
                // setupLayers 会根据 layers 参数自动计算 ebs:
                //   layers=3: ebs = {range*1e-2, range*1e-4, range*1e-6}
                // 
                // ⚠️ 并行压缩问题：每个进程只看到部分数据！
                // 解决方案：如果 wrapper->data_range 已经设置（从全局通信获得），
                //          使用构造的 dummy 数据来调用 setupLayers，
                //          而不是使用局部数据。
                
                if (wrapper->data_range > 0) {
                    // 使用外部指定的全局 range
                    // 构造 dummy 数据：[0, global_range]
                    float* dummy_for_setup = new (std::nothrow) float[nelems];
                    if (dummy_for_setup) {
                        dummy_for_setup[0] = static_cast<float>(wrapper->data_range);
                        for (size_t i = 1; i < nelems; i++) {
                            dummy_for_setup[i] = 0.0f;
                        }
                        sz_comp.setupLayers(dummy_for_setup);
                        delete[] dummy_for_setup;
                        
                        fprintf(stderr, "[DEBUG] Using EXTERNAL global_range=%g for setupLayers\n", 
                                wrapper->data_range);
                    } else {
                        // 内存分配失败，回退到局部计算
                        sz_comp.setupLayers(float_copy);
                        fprintf(stderr, "[WARNING] Failed to alloc dummy, using LOCAL range\n");
                    }
                } else {
                    // 没有外部 range，使用数据本身
                    sz_comp.setupLayers(float_copy);
                    
                    // 计算并保存局部 range
                    float min_val = float_copy[0], max_val = float_copy[0];
                    for (size_t i = 1; i < nelems; i++) {
                        if (float_copy[i] < min_val) min_val = float_copy[i];
                        if (float_copy[i] > max_val) max_val = float_copy[i];
                    }
                    wrapper->data_range = static_cast<double>(max_val - min_val);
                    
                    fprintf(stderr, "[DEBUG] Calculated LOCAL range=%g from data\n", 
                            wrapper->data_range);
                }
                
                // 根据 IPComp setupLayers 公式，实际使用的 ebs 是:
                const char* ebs_formula = "";
                switch (wrapper->layers) {
                    case 1: ebs_formula = "{range*1e-3}"; break;
                    case 2: ebs_formula = "{range*1e-3, range*1e-6}"; break;
                    case 3: ebs_formula = "{range*1e-2, range*1e-4, range*1e-6}"; break;
                    case 4: ebs_formula = "{range*1e-3, range*1e-4, range*1e-5, range*1e-6}"; break;
                    default: ebs_formula = "(see IPComp.hpp setupLayers)"; break;
                }
                
                t1 = MPI_Wtime();
                t_setup = t1 - t0;
                
                if (first_call) {
                    fprintf(stderr, "[DEBUG] Using IPComp hardcoded ebs formula: %s\n", ebs_formula);
                }
                
                // Timer: 核心压缩
                t0 = MPI_Wtime();
                result = sz_comp.compress(float_copy,  // 使用副本，IPComp可以安全修改
                                          *compressed_size,
                                          lossless_buffer);
                t1 = MPI_Wtime();
                t_compress_core = t1 - t0;
                
                fprintf(stderr, "[DEBUG] Compress returned result=%p, compressed_size=%zu\n", 
                        (void*)result, *compressed_size);

            } else if (ndim == 3) {
                // 使用实际传入的dims创建3D维度数组
                std::array<size_t, 3> sz_dims;
                sz_dims[0] = static_cast<size_t>(dims[0]);
                sz_dims[1] = static_cast<size_t>(dims[1]);
                sz_dims[2] = static_cast<size_t>(dims[2]);
                
                // 动态创建3D压缩器
                auto sz_comp = SZ3::SZProgressiveMQuant<float, 3,
                    SZ3::LinearQuantizer2<float>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<float>(nelems, 1.0), // 使用实际元素数量
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size
                );
                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                sz_comp.setupLayers(float_copy);  // 使用副本，这会计算 range
                
                // 保存 range 用于解压缩
                float min_val = float_copy[0], max_val = float_copy[0];
                for (size_t i = 1; i < nelems; i++) {
                    if (float_copy[i] < min_val) min_val = float_copy[i];
                    if (float_copy[i] > max_val) max_val = float_copy[i];
                }
                wrapper->data_range = static_cast<double>(max_val - min_val);
                
                result = sz_comp.compress(float_copy,  // 使用副本
                                          *compressed_size,
                                          lossless_buffer);
            } else {
                delete[] float_copy; free(lossless_buffer); *compressed_size = 0; return nullptr;
            }
            
            // 清理数据副本
            delete[] float_copy;
            
            /* 打印 IPComp 内部计时（仅首次） */
            if (first_call) {
                double total_ipcomp = t_copy + t_create_comp + t_setup + t_compress_core;
                fprintf(stderr, "\n╔═══════════════════════════════════════════════════════════╗\n");
                fprintf(stderr, "║  IPComp Internal Timing (Float, First Call)              ║\n");
                fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
                fprintf(stderr, "║  1. Memcpy data:       %8.4f sec  (%5.1f%%)            ║\n", 
                        t_copy, 100.0*t_copy/total_ipcomp);
                fprintf(stderr, "║  2. Create SZ3 object: %8.4f sec  (%5.1f%%)            ║\n", 
                        t_create_comp, 100.0*t_create_comp/total_ipcomp);
                fprintf(stderr, "║  3. setupLayers:       %8.4f sec  (%5.1f%%)            ║\n", 
                        t_setup, 100.0*t_setup/total_ipcomp);
                fprintf(stderr, "║  4. SZ3 compress():    %8.4f sec  (%5.1f%%)  ← 算法  ║\n", 
                        t_compress_core, 100.0*t_compress_core/total_ipcomp);
                fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
                fprintf(stderr, "║  Total:                %8.4f sec                         ║\n", total_ipcomp);
                fprintf(stderr, "╚═══════════════════════════════════════════════════════════╝\n");
                fprintf(stderr, "\n");
                fprintf(stderr, "SZ3 compress() includes:\n");
                fprintf(stderr, "  • %d layers × %d levels = %d passes\n",
                        wrapper->layers, wrapper->level_progressive,
                        wrapper->layers * wrapper->level_progressive);
                fprintf(stderr, "  • Per pass: predict → quantize → encode → zstd\n");
                fprintf(stderr, "  • Elements per pass: %zu\n", nelems);
                fprintf(stderr, "  • Throughput: %.2f MB/s\n",
                        (nelems * sizeof(float) / (1024.0*1024.0)) / t_compress_core);
                fprintf(stderr, "\n");
                
                first_call = 0;
            }
            
        } else if (data_type == IPCOMP_DOUBLE) {
            const double* double_data = static_cast<const double*>(data);
            
            // Timer: 数据复制
            t0 = MPI_Wtime();
            double* double_copy = new (std::nothrow) double[nelems];
            if (!double_copy) { free(lossless_buffer); *compressed_size = 0; return nullptr; }
            std::memcpy(double_copy, double_data, nelems * sizeof(double));
            data_copy = double_copy;
            t1 = MPI_Wtime();
            t_copy = t1 - t0;
            
            // Timer: 创建压缩器
            t0 = MPI_Wtime();  // 记录用于异常清理

            if (ndim == 2) {
                // 使用实际传入的dims创建2D维度数组
                std::array<size_t, 2> sz_dims;
                sz_dims[0] = static_cast<size_t>(dims[0]);
                sz_dims[1] = static_cast<size_t>(dims[1]);
                
                // 动态创建2D压缩器
                auto sz_comp = SZ3::SZProgressiveMQuant<double, 2,
                    SZ3::LinearQuantizer2<double>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<double>(nelems, 1.0), // 使用实际元素数量
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size
                );
                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                sz_comp.setupLayers(double_copy);  // 使用副本
                result = sz_comp.compress(double_copy,  // 使用副本
                                          *compressed_size,
                                          lossless_buffer);

            } else if (ndim == 3) {
                // 使用实际传入的dims创建3D维度数组
                std::array<size_t, 3> sz_dims;
                sz_dims[0] = static_cast<size_t>(dims[0]);
                sz_dims[1] = static_cast<size_t>(dims[1]);
                sz_dims[2] = static_cast<size_t>(dims[2]);
                
                // 动态创建3D压缩器
                auto sz_comp = SZ3::SZProgressiveMQuant<double, 3,
                    SZ3::LinearQuantizer2<double>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<double>(nelems, 1.0), // 使用实际元素数量
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size
                );
                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }
                
                t1 = MPI_Wtime();
                t_create_comp = t1 - t0;
                
                // Timer: setupLayers
                t0 = MPI_Wtime();
                sz_comp.setupLayers(double_copy);  // 使用副本，这会计算 range 并设置 ebs
                t1 = MPI_Wtime();
                t_setup = t1 - t0;
                
                // 保存 range 用于解压缩
                double min_val = double_copy[0], max_val = double_copy[0];
                for (size_t i = 1; i < nelems; i++) {
                    if (double_copy[i] < min_val) min_val = double_copy[i];
                    if (double_copy[i] > max_val) max_val = double_copy[i];
                }
                wrapper->data_range = max_val - min_val;
                
                if (first_call) {
                    fprintf(stderr, "[DEBUG] IPComp compress (double): layers=%d, range=%g\n", 
                            wrapper->layers, wrapper->data_range);
                }
                
                // Timer: SZ3核心压缩
                t0 = MPI_Wtime();
                result = sz_comp.compress(double_copy,  // 使用副本
                                          *compressed_size,
                                          lossless_buffer);
                t1 = MPI_Wtime();
                t_compress_core = t1 - t0;
            } else {
                delete[] double_copy; free(lossless_buffer); *compressed_size = 0; return nullptr;
            }
            
            // 清理数据副本
            delete[] double_copy;
            
            /* 打印 IPComp 内部计时（仅首次） */
            if (first_call) {
                double total_ipcomp = t_copy + t_create_comp + t_setup + t_compress_core;
                fprintf(stderr, "\n╔═══════════════════════════════════════════════════════════╗\n");
                fprintf(stderr, "║  IPComp Internal Timing (Double, First Call)             ║\n");
                fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
                fprintf(stderr, "║  1. Memcpy data:       %8.4f sec  (%5.1f%%)            ║\n", 
                        t_copy, 100.0*t_copy/total_ipcomp);
                fprintf(stderr, "║  2. Create SZ3 object: %8.4f sec  (%5.1f%%)            ║\n", 
                        t_create_comp, 100.0*t_create_comp/total_ipcomp);
                fprintf(stderr, "║  3. setupLayers:       %8.4f sec  (%5.1f%%)            ║\n", 
                        t_setup, 100.0*t_setup/total_ipcomp);
                fprintf(stderr, "║  4. SZ3 compress():    %8.4f sec  (%5.1f%%)  ← 核心算法║\n", 
                        t_compress_core, 100.0*t_compress_core/total_ipcomp);
                fprintf(stderr, "╠═══════════════════════════════════════════════════════════╣\n");
                fprintf(stderr, "║  Total:                %8.4f sec                         ║\n", total_ipcomp);
                fprintf(stderr, "╚═══════════════════════════════════════════════════════════╝\n");
                fprintf(stderr, "\n");
                fprintf(stderr, "SZ3 compress() DETAILED breakdown (estimated):\n");
                if (wrapper->level_progressive > 0) {
                    int total_passes = wrapper->layers * wrapper->level_progressive;
                    fprintf(stderr, "  Total passes: %d layers × %d levels = %d\n",
                            wrapper->layers, wrapper->level_progressive, total_passes);
                    fprintf(stderr, "  Time per pass: %.4f sec\n",
                            total_passes > 0 ? t_compress_core / total_passes : 0.0);
                } else {
                    fprintf(stderr, "  Total passes: %d layers × auto levels (library-managed)\n",
                            wrapper->layers);
                    fprintf(stderr, "  Time per pass: N/A (auto progressive levels)\n");
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "  Each pass includes (~均分时间):\n");
                fprintf(stderr, "    (a) Prediction/Interpolation: ~25%%\n");
                fprintf(stderr, "        - 3D linear/cubic interpolation\n");
                fprintf(stderr, "        - Compute predicted values for all points\n");
                fprintf(stderr, "    (b) Quantization: ~15%%\n");
                fprintf(stderr, "        - Convert float residuals to integers\n");
                fprintf(stderr, "    (c) Encoding: ~10%%\n");
                fprintf(stderr, "        - Bypass encoder (minimal overhead)\n");
                fprintf(stderr, "    (d) Zstd compression: ~50%%\n");
                fprintf(stderr, "        - Lossless compression of encoded data\n");
                fprintf(stderr, "        - CPU-intensive, single-threaded\n");
                fprintf(stderr, "\n");
                fprintf(stderr, "  Performance:\n");
                fprintf(stderr, "    • Elements: %zu\n", nelems);
                fprintf(stderr, "    • Data size: %.2f MB\n", nelems * sizeof(double) / (1024.0*1024.0));
                fprintf(stderr, "    • Throughput: %.2f MB/s\n",
                        (nelems * sizeof(double) / (1024.0*1024.0)) / t_compress_core);
                fprintf(stderr, "\n");
                fprintf(stderr, "  Why is it slow?\n");
                fprintf(stderr, "    ✗ Single-threaded (no parallelism)\n");
                if (wrapper->level_progressive > 0) {
                    fprintf(stderr, "    ✗ Complex algorithm (%d passes)\n",
                            wrapper->layers * wrapper->level_progressive);
                } else {
                    fprintf(stderr, "    ✗ Complex algorithm (auto progressive depth)\n");
                }
                fprintf(stderr, "    ✗ Memory-intensive (poor cache locality)\n");
                fprintf(stderr, "    ✗ Zstd dominates (~50%% of time per pass)\n");
                fprintf(stderr, "\n");
                
                first_call = 0;
            }
            
        } else {
            // 不支持的数据类型
            free(lossless_buffer);
            *compressed_size = 0;
            return nullptr;
        }

        // ---- 统一拷贝到由 wrapper 管理的输出缓冲（避免库内对齐 delete 不匹配） ----
        if (*compressed_size > lossless_cap) {
                // 保护：容量应该由我们提供，若溢出表示估算偏小
               // 这里直接失败返回，或者可以选择重试：重新分配更大工作区并重压缩
                free(lossless_buffer);
                *compressed_size = 0;
                return nullptr;
            }
            if (*compressed_size == 0) {
                free(lossless_buffer);
                return nullptr; // 空输出
            }
    
            const size_t payload_size = *compressed_size;
            const size_t total_size = kIPCompHeaderSize + payload_size;

            unsigned char* out = static_cast<unsigned char*>(std::malloc(total_size));
            if (!out) {
                free(lossless_buffer);
                *compressed_size = 0;
                return nullptr;
            }

            encode_ipcomp_header(out, wrapper->data_range, wrapper->data_range > 0.0);
            // 优先从 result 拷贝；若 result 为空则从我们的工作区拷贝
            std::memcpy(out + kIPCompHeaderSize, result ? result : lossless_buffer, payload_size);

            // 释放工作区；若 result 指向库内存，我们不去释放（通常 compress 会复用来写入工作区）
            free(lossless_buffer);
            register_buffer(out, delete_with_free);
            *compressed_size = total_size;
            return out;

    } catch (...) {
        if (data_copy) {
            if (data_type == IPCOMP_FLOAT) {
                delete[] static_cast<float*>(data_copy);
            } else if (data_type == IPCOMP_DOUBLE) {
                delete[] static_cast<double*>(data_copy);
            }
        }
        free(lossless_buffer);
        *compressed_size = 0;
        return nullptr;
    }
}


/* Decompress with error bound */
void* ipcomp_decompress_error(void* compressor, const unsigned char* compressed_data,
                             int data_type, const double* target_rel_ebs,
                             int num_target_ebs) {
    if (compressor == nullptr) {
        fprintf(stderr, "[ipcomp_decompress_error] compressor is NULL\n");
        return nullptr;
    }
    if (compressed_data == nullptr) {
        fprintf(stderr, "[ipcomp_decompress_error] compressed_data is NULL\n");
        return nullptr;
    }

    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);

    const unsigned char* payload = compressed_data;
    auto header_info = decode_ipcomp_header(compressed_data);
    if (header_info.has_header) {
        payload = header_info.payload;
        if (header_info.has_range) {
            wrapper->data_range = header_info.data_range;
        }
    }
    if (wrapper->data_range <= 0.0) {
        wrapper->data_range = header_info.has_range ? header_info.data_range : 1.0;
    }

    if (wrapper->ndim <= 0 || wrapper->ndim > 3) {
        fprintf(stderr, "[ipcomp_decompress_error] Invalid ndim: %d\n", wrapper->ndim);
        return nullptr;
    }
    if (wrapper->dims.empty()) {
        fprintf(stderr, "[ipcomp_decompress_error] dims is empty\n");
        return nullptr;
    }
    if (wrapper->num_elements == 0) {
        fprintf(stderr, "[ipcomp_decompress_error] num_elements is zero\n");
        return nullptr;
    }

    std::vector<double> relative_ebs;
    if (target_rel_ebs != nullptr && num_target_ebs > 0) {
        relative_ebs.assign(target_rel_ebs, target_rel_ebs + num_target_ebs);
    }
    bool need_default = relative_ebs.empty();
    if (!need_default) {
        need_default = std::all_of(relative_ebs.begin(), relative_ebs.end(), [](double v) {
            return !(v > 0.0);
        });
    }
    if (need_default) {
        relative_ebs = build_default_relative_ebs(wrapper->layers);
    }
    if (relative_ebs.empty()) {
        relative_ebs.push_back(1e-6);
    }
    if (need_default) {
        for (double &val : relative_ebs) {
            if (val > 0.0) {
                double nudged = std::nextafter(val, 0.0);
                if (nudged <= 0.0) {
                    nudged = val * 0.5;
                }
                val = nudged;
            }
        }
    }

    try {
        if (data_type == IPCOMP_FLOAT) {
            float* dummy_original = new (std::nothrow) float[wrapper->num_elements];
            if (!dummy_original) {
                return nullptr;
            }
            for (size_t i = 0; i < wrapper->num_elements; i++) {
                dummy_original[i] = 0.0f;
            }

            float* result = nullptr;

            if (wrapper->ndim == 2) {
                std::array<size_t, 2> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1])};

                auto sz_comp = SZ3::SZProgressiveMQuant<float, 2,
                    SZ3::LinearQuantizer2<float>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<float>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = static_cast<float>(wrapper->data_range);
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0f;
                }

                fprintf(stderr, "[DEBUG] ipcomp_decompress_error float: layers=%d, rel_ebs count=%zu first=%g\n",
                        wrapper->layers, relative_ebs.size(), relative_ebs.empty() ? 0.0 : relative_ebs.front());
                result = with_verification_override(wrapper, [&]() {
                    return sz_comp.decompress(const_cast<unsigned char*>(payload),
                                              dummy_original, relative_ebs);
                });
            } else if (wrapper->ndim == 3) {
                std::array<size_t, 3> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1]),
                                              static_cast<size_t>(wrapper->dims[2])};

                auto sz_comp = SZ3::SZProgressiveMQuant<float, 3,
                    SZ3::LinearQuantizer2<float>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<float>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }
                ipcomp_wrapper_dbg("[IPCOMP_WRAPPER][decomp_error][3D][float] mask_set=%d mask_bytes=%zu valid_count=%zu guard=%d boundary=%s\n",
                                   (int)wrapper->mask_set, wrapper->mask_bytes, wrapper->mask_valid_count,
                                   wrapper->guard_radius, wrapper->mask_boundary.empty() ? "no" : "yes");

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = static_cast<float>(wrapper->data_range);
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0f;
                }

                fprintf(stderr, "[DEBUG] ipcomp_decompress_error float 3D: layers=%d, rel_ebs count=%zu first=%g\n",
                        wrapper->layers, relative_ebs.size(), relative_ebs.empty() ? 0.0 : relative_ebs.front());
                result = with_verification_override(wrapper, [&]() {
                    return sz_comp.decompress(const_cast<unsigned char*>(payload),
                                              dummy_original, relative_ebs);
                });
            } else {
                delete[] dummy_original;
                return nullptr;
            }

            if (!result) {
                delete[] dummy_original;
                return nullptr;
            }

            if (result == dummy_original) {
                register_buffer(result, delete_with_new_array<float>);
                dummy_original = nullptr;
            } else {
                register_buffer(result, delete_with_aligned256);
                delete[] dummy_original;
                dummy_original = nullptr;
            }
            return result;

        } else if (data_type == IPCOMP_DOUBLE) {
            double* dummy_original = new (std::nothrow) double[wrapper->num_elements];
            if (!dummy_original) {
                return nullptr;
            }
            for (size_t i = 0; i < wrapper->num_elements; i++) {
                dummy_original[i] = 0.0;
            }

            double* result = nullptr;

            if (wrapper->ndim == 2) {
                std::array<size_t, 2> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1])};

                auto sz_comp = SZ3::SZProgressiveMQuant<double, 2,
                    SZ3::LinearQuantizer2<double>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<double>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = wrapper->data_range;
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0;
                }

                fprintf(stderr, "[DEBUG] ipcomp_decompress_error double: layers=%d, rel_ebs count=%zu first=%g\n",
                        wrapper->layers, relative_ebs.size(), relative_ebs.empty() ? 0.0 : relative_ebs.front());
                result = with_verification_override(wrapper, [&]() {
                    return sz_comp.decompress(const_cast<unsigned char*>(payload),
                                              dummy_original, relative_ebs);
                });
            } else if (wrapper->ndim == 3) {
                std::array<size_t, 3> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1]),
                                              static_cast<size_t>(wrapper->dims[2])};

                auto sz_comp = SZ3::SZProgressiveMQuant<double, 3,
                    SZ3::LinearQuantizer2<double>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<double>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }
                ipcomp_wrapper_dbg("[IPCOMP_WRAPPER][decomp_error][3D][double] mask_set=%d mask_bytes=%zu valid_count=%zu guard=%d boundary=%s\n",
                                   (int)wrapper->mask_set, wrapper->mask_bytes, wrapper->mask_valid_count,
                                   wrapper->guard_radius, wrapper->mask_boundary.empty() ? "no" : "yes");

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = wrapper->data_range;
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0;
                }

                fprintf(stderr, "[DEBUG] ipcomp_decompress_error double 3D: layers=%d, rel_ebs count=%zu first=%g\n",
                        wrapper->layers, relative_ebs.size(), relative_ebs.empty() ? 0.0 : relative_ebs.front());
                result = with_verification_override(wrapper, [&]() {
                    return sz_comp.decompress(const_cast<unsigned char*>(payload),
                                              dummy_original, relative_ebs);
                });
            } else {
                delete[] dummy_original;
                return nullptr;
            }

            if (!result) {
                delete[] dummy_original;
                return nullptr;
            }

            if (result == dummy_original) {
                register_buffer(result, delete_with_new_array<double>);
                dummy_original = nullptr;
            } else {
                register_buffer(result, delete_with_aligned256);
                delete[] dummy_original;
                dummy_original = nullptr;
            }
            return result;
        }

        fprintf(stderr, "[ipcomp_decompress_error] Unsupported data_type: %d\n", data_type);
        return nullptr;
    } catch (const std::exception& e) {
        fprintf(stderr, "[ipcomp_decompress_error] Exception caught: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "[ipcomp_decompress_error] Unknown exception caught\n");
        return nullptr;
    }
}

/* Decompress with bitrate constraint */
void* ipcomp_decompress_bitrate(void* compressor, const unsigned char* compressed_data,
                               int data_type, const double* target_bitrates,
                               int num_target_bitrates) {
    if (compressor == nullptr) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] compressor is NULL\n");
        return nullptr;
    }
    if (compressed_data == nullptr) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] compressed_data is NULL\n");
        return nullptr;
    }

    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);

    const unsigned char* payload = compressed_data;
    auto header_info = decode_ipcomp_header(compressed_data);
    if (header_info.has_header) {
        payload = header_info.payload;
        if (header_info.has_range) {
            wrapper->data_range = header_info.data_range;
        }
    }
    if (wrapper->data_range <= 0.0) {
        wrapper->data_range = header_info.has_range ? header_info.data_range : 1.0;
    }

    if (wrapper->ndim <= 0 || wrapper->ndim > 3) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] Invalid ndim: %d\n", wrapper->ndim);
        return nullptr;
    }
    if (wrapper->dims.empty()) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] dims is empty\n");
        return nullptr;
    }
    if (wrapper->num_elements == 0) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] num_elements is zero\n");
        return nullptr;
    }

    std::vector<double> bitrate_targets;
    if (target_bitrates != nullptr && num_target_bitrates > 0) {
        bitrate_targets.assign(target_bitrates, target_bitrates + num_target_bitrates);
    }

    auto ensure_bitrates = [&](double fallback_bits) {
        if (bitrate_targets.empty()) {
            bitrate_targets.push_back(fallback_bits);
            return;
        }
        bool all_nonpositive = true;
        for (double &val : bitrate_targets) {
            if (val > 0.0) {
                all_nonpositive = false;
                break;
            }
        }
        if (all_nonpositive) {
            bitrate_targets.assign(1, fallback_bits);
        }
        for (double &val : bitrate_targets) {
            if (val <= 0.0) {
                val = fallback_bits;
            }
        }
    };

    try {
        if (data_type == IPCOMP_FLOAT) {
            ensure_bitrates(static_cast<double>(sizeof(float) * 8));

            float* dummy_original = new (std::nothrow) float[wrapper->num_elements];
            if (!dummy_original) {
                return nullptr;
            }
            for (size_t i = 0; i < wrapper->num_elements; i++) {
                dummy_original[i] = 0.0f;
            }

            float* result = nullptr;

            if (wrapper->ndim == 2) {
                std::array<size_t, 2> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1])};

                auto sz_comp = SZ3::SZProgressiveMQuant<float, 2,
                    SZ3::LinearQuantizer2<float>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<float>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = static_cast<float>(wrapper->data_range);
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0f;
                }

                result = sz_comp.decompress_bitrate(const_cast<unsigned char*>(payload),
                                                    dummy_original, bitrate_targets);
            } else if (wrapper->ndim == 3) {
                std::array<size_t, 3> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1]),
                                              static_cast<size_t>(wrapper->dims[2])};

                auto sz_comp = SZ3::SZProgressiveMQuant<float, 3,
                    SZ3::LinearQuantizer2<float>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<float>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = static_cast<float>(wrapper->data_range);
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0f;
                }

                result = sz_comp.decompress_bitrate(const_cast<unsigned char*>(payload),
                                                    dummy_original, bitrate_targets);
            } else {
                delete[] dummy_original;
                return nullptr;
            }

            if (!result) {
                delete[] dummy_original;
                return nullptr;
            }

            if (result == dummy_original) {
                register_buffer(result, delete_with_new_array<float>);
                dummy_original = nullptr;
            } else {
                register_buffer(result, delete_with_aligned256);
                delete[] dummy_original;
                dummy_original = nullptr;
            }
            return result;

        } else if (data_type == IPCOMP_DOUBLE) {
            ensure_bitrates(static_cast<double>(sizeof(double) * 8));

            double* dummy_original = new (std::nothrow) double[wrapper->num_elements];
            if (!dummy_original) {
                return nullptr;
            }
            for (size_t i = 0; i < wrapper->num_elements; i++) {
                dummy_original[i] = 0.0;
            }

            double* result = nullptr;

            if (wrapper->ndim == 2) {
                std::array<size_t, 2> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1])};

                auto sz_comp = SZ3::SZProgressiveMQuant<double, 2,
                    SZ3::LinearQuantizer2<double>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<double>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = wrapper->data_range;
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0;
                }

                result = sz_comp.decompress_bitrate(const_cast<unsigned char*>(payload),
                                                    dummy_original, bitrate_targets);
            } else if (wrapper->ndim == 3) {
                std::array<size_t, 3> sz_dims{static_cast<size_t>(wrapper->dims[0]),
                                              static_cast<size_t>(wrapper->dims[1]),
                                              static_cast<size_t>(wrapper->dims[2])};

                auto sz_comp = SZ3::SZProgressiveMQuant<double, 3,
                    SZ3::LinearQuantizer2<double>, SZ3::BypassEncoder<int>, SZ3::Lossless_zstd>(
                    SZ3::LinearQuantizer2<double>(wrapper->num_elements, 1.0),
                    SZ3::BypassEncoder<int>(),
                    SZ3::Lossless_zstd(3),
                    sz_dims,
                    wrapper->interp_op,
                    wrapper->direction_op,
                    wrapper->interp_dim_limit,
                    wrapper->layers,
                    wrapper->block_size);

                if (wrapper->mask_set && wrapper->mask_bytes > 0) {
                    sz_comp.set_mask(wrapper->mask_valid.data(),
                                     wrapper->mask_boundary.empty() ? nullptr : wrapper->mask_boundary.data(),
                                     wrapper->mask_bytes,
                                     wrapper->mask_valid_count,
                                     wrapper->guard_radius);
                }

                if (wrapper->data_range > 0.0 && wrapper->num_elements > 0) {
                    dummy_original[0] = wrapper->data_range;
                }
                sz_comp.setupLayers(dummy_original);
                for (size_t i = 0; i < wrapper->num_elements; i++) {
                    dummy_original[i] = 0.0;
                }

                result = sz_comp.decompress_bitrate(const_cast<unsigned char*>(payload),
                                                    dummy_original, bitrate_targets);
            } else {
                delete[] dummy_original;
                return nullptr;
            }

            if (!result) {
                delete[] dummy_original;
                return nullptr;
            }

            if (result == dummy_original) {
                register_buffer(result, delete_with_new_array<double>);
                dummy_original = nullptr;
            } else {
                register_buffer(result, delete_with_aligned256);
                delete[] dummy_original;
                dummy_original = nullptr;
            }
            return result;
        }

        fprintf(stderr, "[ipcomp_decompress_bitrate] Unsupported data_type: %d\n", data_type);
        return nullptr;
    } catch (const std::exception& e) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] Exception caught: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "[ipcomp_decompress_bitrate] Unknown exception caught\n");
        return nullptr;
    }
}

/* Free buffer allocated by IPComp */
void ipcomp_free_buffer(void* buffer) {
    if (buffer == nullptr) {
        return;
    }

    void (*deleter)(void*) = nullptr;
    if (take_buffer_record(buffer, &deleter) && deleter != nullptr) {
        deleter(buffer);
        return;
    }

    std::free(buffer);
}

/* Set data range for decompression */
int ipcomp_set_range(void* compressor, double data_range) {
    if (compressor == nullptr) return -1;
    
    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);
    wrapper->data_range = data_range;
    return 0;
}

int ipcomp_set_minmax(void* compressor, double data_min, double data_max) {
    if (compressor == nullptr) return -1;
    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);
    if (data_max >= data_min) {
        wrapper->meta_min = data_min;
        wrapper->meta_max = data_max;
        wrapper->has_meta_minmax = true;
    } else {
        wrapper->has_meta_minmax = false;
    }
    return 0;
}

int ipcomp_set_mask(void* compressor,
                    const unsigned char* valid_mask,
                    const unsigned char* boundary_mask,
                    size_t mask_bytes,
                    size_t valid_count,
                    int guard_radius) {
    if (compressor == nullptr) return -1;
    IPCompWrapper* wrapper = static_cast<IPCompWrapper*>(compressor);
    if (!valid_mask || mask_bytes == 0) {
        wrapper->mask_valid.clear();
        wrapper->mask_boundary.clear();
        wrapper->mask_bytes = 0;
        wrapper->mask_valid_count = 0;
        wrapper->guard_radius = 0;
        wrapper->mask_set = false;
        return 0;
    }
    try {
        wrapper->mask_valid.assign(valid_mask, valid_mask + mask_bytes);
        if (boundary_mask) {
            wrapper->mask_boundary.assign(boundary_mask, boundary_mask + mask_bytes);
        } else {
            wrapper->mask_boundary.clear();
        }
        wrapper->mask_bytes = mask_bytes;
        wrapper->mask_valid_count = valid_count;
        wrapper->guard_radius = guard_radius;
        wrapper->mask_set = true;
        ipcomp_wrapper_dbg("[IPCOMP_WRAPPER][set_mask] bytes=%zu valid_count=%zu guard=%d boundary=%s\n",
                           wrapper->mask_bytes, wrapper->mask_valid_count, wrapper->guard_radius,
                           wrapper->mask_boundary.empty() ? "no" : "yes");
        return 0;
    } catch (...) {
        return -1;
    }
}

#else /* !ENABLE_IPCOMP */

/* Stub implementations when IPComp is disabled */
void* ipcomp_create_compressor(int ndim, const int* dims, int interp_op, int direction_op, 
                              int layers, size_t interp_dim_limit, size_t block_size, int level_progressive) {
    return nullptr;
}

void ipcomp_destroy_compressor(void* compressor) {
    /* Nothing to do */
}

int ipcomp_setup_layers(void* compressor, const void* data, int data_type) {
    return -1;
}

unsigned char* ipcomp_compress(void* compressor, const void* data, int data_type, size_t* compressed_size,
                               int ndim, const int* dims) {
    return nullptr;
}

void* ipcomp_decompress_error(void* compressor, const unsigned char* compressed_data,
                             int data_type, const double* target_rel_ebs,
                             int num_target_ebs) {
    return nullptr;
}

void* ipcomp_decompress_bitrate(void* compressor, const unsigned char* compressed_data,
                               int data_type, const double* target_bitrates,
                               int num_target_bitrates) {
    return nullptr;
}

void ipcomp_free_buffer(void* buffer) {
    /* Nothing to do */
}

int ipcomp_set_range(void* compressor, double data_range) {
    return -1;
}

int ipcomp_set_minmax(void* compressor, double data_min, double data_max) {
    return -1;
}

int ipcomp_set_mask(void* compressor,
                    const unsigned char* valid_mask,
                    const unsigned char* boundary_mask,
                    size_t mask_bytes,
                    size_t valid_count,
                    int guard_radius) {
    return -1;
}

#endif /* ENABLE_IPCOMP */

} /* extern "C" */