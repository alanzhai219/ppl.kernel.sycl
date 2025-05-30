#ifndef PPL_KERNEL_SYCL_REFORMAT_REFORMAT_H_
#define PPL_KERNEL_SYCL_REFORMAT_REFORMAT_H_

#include "../common/device.h"
#include "../common/types.h"

namespace ppl { namespace kernel { namespace sycl { namespace reformat {

// 数据格式转换（NCHW <-> NHWC）
template <typename T>
RetCode ConvertFormat(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape,
    DataFormat src_format,
    DataFormat dst_format);

// 数据类型转换
template <typename SrcType, typename DstType>
RetCode ConvertType(
    Device* device,
    const SrcType* input,
    DstType* output,
    const TensorShape& shape);

}}}} // namespace ppl::kernel::sycl::reformat

#endif // PPL_KERNEL_SYCL_REFORMAT_REFORMAT_H_