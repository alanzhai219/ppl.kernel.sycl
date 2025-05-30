#ifndef PPL_KERNEL_SYCL_UNARY_UNARY_H_
#define PPL_KERNEL_SYCL_UNARY_UNARY_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace unary {

// 绝对值运算
template <typename T>
RetCode Abs(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

// 负值运算
template <typename T>
RetCode Neg(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

// 指数运算
template <typename T>
RetCode Exp(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

// 对数运算
template <typename T>
RetCode Log(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

// 平方根运算
template <typename T>
RetCode Sqrt(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

}}}} // namespace ppl::kernel::sycl::unary

#endif // PPL_KERNEL_SYCL_UNARY_UNARY_H_