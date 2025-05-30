#ifndef PPL_KERNEL_SYCL_REDUCE_REDUCE_H_
#define PPL_KERNEL_SYCL_REDUCE_REDUCE_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace reduce {

// 归约操作类型
enum class ReduceOp {
    SUM = 0,
    MAX,
    MIN,
    PROD,
    MEAN
};

// 沿指定轴进行归约操作
template <typename T>
RetCode Reduce(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& input_shape,
    const TensorShape& output_shape,
    const int* axes,
    int num_axes,
    ReduceOp op);

}}}} // namespace ppl::kernel::sycl::reduce

#endif // PPL_KERNEL_SYCL_REDUCE_REDUCE_H_