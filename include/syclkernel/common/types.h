#ifndef PPL_KERNEL_SYCL_COMMON_TYPES_H_
#define PPL_KERNEL_SYCL_COMMON_TYPES_H_

#include <sycl/sycl.hpp>
#include <cstdint>

namespace ppl { namespace kernel { namespace sycl {

// 基本数据类型定义
using int8_t = std::int8_t;
using int16_t = std::int16_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;
using uint8_t = std::uint8_t;
using uint16_t = std::uint16_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;
using float16_t = ::sycl::half;
using float32_t = float;
using float64_t = double;

// 错误码定义
enum class RetCode {
    SUCCESS = 0,
    INVALID_VALUE = 1,
    UNSUPPORTED = 2,
    OUT_OF_MEMORY = 3,
    INTERNAL_ERROR = 4,
    DEVICE_ERROR = 5,
    NOT_IMPLEMENTED = 6
};

// 数据类型枚举
enum class DataType {
    UNKNOWN = 0,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    BOOL,
    COMPLEX64,
    COMPLEX128
};

// 数据格式枚举
enum class DataFormat {
    UNKNOWN = 0,
    NDARRAY,
    NHWC,
    NCHW
};

// 张量形状
struct TensorShape {
    int64_t dims[8];
    uint32_t dim_count;
    
    TensorShape() : dim_count(0) {
        for (int i = 0; i < 8; ++i) {
            dims[i] = 1;
        }
    }
    
    int64_t GetElementCount() const {
        int64_t count = 1;
        for (uint32_t i = 0; i < dim_count; ++i) {
            count *= dims[i];
        }
        return count;
    }
};

}}} // namespace ppl::kernel::sycl

#endif // PPL_KERNEL_SYCL_COMMON_TYPES_H_