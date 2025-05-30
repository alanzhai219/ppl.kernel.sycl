#ifndef PPL_KERNEL_SYCL_NN_GELU_H_
#define PPL_KERNEL_SYCL_NN_GELU_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace nn {

// GELU激活函数
template <typename T>
RetCode GELU(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

}}}} // namespace ppl::kernel::sycl::nn

#endif // PPL_KERNEL_SYCL_NN_GELU_H_