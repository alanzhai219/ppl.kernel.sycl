#ifndef PPL_KERNEL_SYCL_ARITHMETIC_ARITHMETIC_H_
#define PPL_KERNEL_SYCL_ARITHMETIC_ARITHMETIC_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace arithmetic {

// 加法运算
template <typename T>
RetCode Add(
    Device* device,
    const T* input_a,
    const T* input_b,
    T* output,
    const TensorShape& shape);

// 减法运算
template <typename T>
RetCode Sub(
    Device* device,
    const T* input_a,
    const T* input_b,
    T* output,
    const TensorShape& shape);

// 乘法运算
template <typename T>
RetCode Mul(
    Device* device,
    const T* input_a,
    const T* input_b,
    T* output,
    const TensorShape& shape);

// 除法运算
template <typename T>
RetCode Div(
    Device* device,
    const T* input_a,
    const T* input_b,
    T* output,
    const TensorShape& shape);

}}}} // namespace ppl::kernel::sycl::arithmetic

#endif // PPL_KERNEL_SYCL_ARITHMETIC_ARITHMETIC_H_