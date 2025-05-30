#ifndef PPL_KERNEL_SYCL_NN_SILU_H_
#define PPL_KERNEL_SYCL_NN_SILU_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace nn {

// SILU激活函数
template <typename T>
RetCode SILU(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape);

}}}} // namespace ppl::kernel::sycl::nn

#endif // PPL_KERNEL_SYCL_NN_SILU_H_