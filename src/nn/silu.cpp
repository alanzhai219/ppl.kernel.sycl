#include "syclkernel/nn/silu.h"
#include <sycl/sycl.hpp>

namespace ppl { namespace kernel { namespace sycl { namespace nn {

template <typename T>
RetCode SILU(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape) {
    
    if (!device || !input || !output) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    try {
        auto& queue = device->GetQueue();
        const int64_t element_count = shape.GetElementCount();
        
        // 获取设备属性
        DeviceProperties prop;
        device->GetDeviceProperties(&prop);
        
        // 计算工作组大小
        const size_t work_group_size = std::min(
            static_cast<size_t>(256), 
            prop.max_work_group_size);
        
        const size_t num_work_groups = (element_count + work_group_size - 1) / work_group_size;
        const size_t global_size = num_work_groups * work_group_size;
        
        // 提交SYCL内核
        queue.submit([&](::sycl::handler& cgh) {
            cgh.parallel_for(
                ::sycl::nd_range<1>(
                    ::sycl::range<1>(global_size),
                    ::sycl::range<1>(work_group_size)
                ),
                [=](::sycl::nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx < element_count) {
                        // SILU: x * sigmoid(x) = x / (1 + exp(-x))
                        T val = input[idx];
                        T out_val = val / (static_cast<T>(1.0) + ::sycl::exp(-val));
                        output[idx] = out_val;
                    }
                }
            );
        });
        
        // 等待完成
        queue.wait();
        
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        // 处理SYCL异常
        return RetCode::INTERNAL_ERROR;
    }
}

// 显式实例化常用类型
template RetCode SILU<float>(Device*, const float*, float*, const TensorShape&);
template RetCode SILU<float16_t>(Device*, const float16_t*, float16_t*, const TensorShape&);

}}}} // namespace ppl::kernel::sycl::nn