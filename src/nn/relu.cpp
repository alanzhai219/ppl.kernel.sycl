#include "syclkernel/nn/relu.h"
#include <sycl/sycl.hpp>

namespace ppl { namespace kernel { namespace sycl { namespace nn {

template <typename T>
RetCode ReLU(
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
                        output[idx] = ::sycl::max(input[idx], static_cast<T>(0));
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
template RetCode ReLU<float>(Device*, const float*, float*, const TensorShape&);
template RetCode ReLU<float16_t>(Device*, const float16_t*, float16_t*, const TensorShape&);

// LeakyReLU实现
template <typename T>
RetCode LeakyReLU(
    Device* device,
    const T* input,
    T* output,
    const TensorShape& shape,
    float alpha) {
    
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
                        // LeakyReLU: f(x) = x if x > 0, f(x) = alpha * x if x <= 0
                        T val = input[idx];
                        output[idx] = val > static_cast<T>(0) ? val : static_cast<T>(alpha * val);
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
template RetCode LeakyReLU<float>(Device*, const float*, float*, const TensorShape&, float);
template RetCode LeakyReLU<float16_t>(Device*, const float16_t*, float16_t*, const TensorShape&, float);

// 如果需要支持int8_t类型，可以添加以下特化实现
template <>
RetCode LeakyReLU<int8_t>(
    Device* device,
    const int8_t* input,
    int8_t* output,
    const TensorShape& shape,
    float alpha) {
    
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
                        // LeakyReLU for int8_t
                        int8_t val = input[idx];
                        output[idx] = val > 0 ? val : static_cast<int8_t>(alpha * val);
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

}}}} // namespace ppl::kernel::sycl::nn