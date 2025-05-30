#ifndef PPL_KERNEL_SYCL_NN_RELU_H_
#define PPL_KERNEL_SYCL_NN_RELU_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace nn {

// ReLU激活函数
template <typename T>
RetCode ReLU(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

// LeakyReLU激活函数
template <typename T>
RetCode LeakyReLU(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape,
    float alpha);

}}}} // namespace ppl::kernel::sycl::nn

#endif // PPL_KERNEL_SYCL_NN_RELU_H_