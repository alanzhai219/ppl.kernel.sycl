#include "syclkernel/common/device.h"
#include <vector>

namespace ppl { namespace kernel { namespace sycl {

class Device::Impl {
public:
    Impl() : initialized_(false), device_type_(DeviceType::ANY) {}
    
    ~Impl() {}
    
    RetCode Init(DeviceType type) {
        try {
            device_type_ = type;
            
            // 根据设备类型选择SYCL设备
            ::sycl::device selected_device;
            
            switch (type) {
                case DeviceType::CPU:
                    selected_device = ::sycl::device(::sycl::cpu_selector_v);
                    break;
                case DeviceType::GPU:
                    selected_device = ::sycl::device(::sycl::gpu_selector_v);
                    break;
                case DeviceType::ACCELERATOR:
                    selected_device = ::sycl::device(::sycl::accelerator_selector_v);
                    break;
                case DeviceType::ANY:
                default:
                    selected_device = ::sycl::device(::sycl::default_selector_v);
                    break;
            }
            
            // 创建SYCL队列
            queue_ = ::sycl::queue(selected_device, ::sycl::property::queue::in_order());
            
            // 设置初始化标志
            initialized_ = true;
            
            return RetCode::SUCCESS;
            
        } catch (const ::sycl::exception& e) {
            initialized_ = false;
            return RetCode::DEVICE_ERROR;
        }
    }
    
    RetCode GetDeviceProperties(DeviceProperties* prop) const {
        if (!initialized_ || !prop) {
            return RetCode::INVALID_VALUE;
        }
        
        try {
            auto device = queue_.get_device();
            
            // 填充设备属性
            prop->name = device.get_info<::sycl::info::device::name>();
            
            // 设置设备类型
            if (device.is_cpu()) {
                prop->type = DeviceType::CPU;
            } else if (device.is_gpu()) {
                prop->type = DeviceType::GPU;
            } else if (device.is_accelerator()) {
                prop->type = DeviceType::ACCELERATOR;
            } else {
                prop->type = DeviceType::ANY;
            }
            
            // 获取设备能力
            prop->max_work_group_size = device.get_info<::sycl::info::device::max_work_group_size>();
            prop->max_compute_units = device.get_info<::sycl::info::device::max_compute_units>();
            prop->global_mem_size = device.get_info<::sycl::info::device::global_mem_size>();
            prop->local_mem_size = device.get_info<::sycl::info::device::local_mem_size>();
            prop->max_mem_alloc_size = device.get_info<::sycl::info::device::max_mem_alloc_size>();
            prop->has_unified_memory = device.has(::sycl::aspect::usm_shared_allocations);
            
            return RetCode::SUCCESS;
            
        } catch (const ::sycl::exception& e) {
            return RetCode::DEVICE_ERROR;
        }
    }
    
    ::sycl::queue& GetQueue() {
        return queue_;
    }
    
    RetCode Synchronize() {
        if (!initialized_) {
            return RetCode::DEVICE_ERROR;
        }
        
        try {
            queue_.wait_and_throw();
            return RetCode::SUCCESS;
        } catch (const ::sycl::exception& e) {
            return RetCode::DEVICE_ERROR;
        }
    }
    
    DeviceType GetDeviceType() const {
        return device_type_;
    }
    
    bool IsInitialized() const {
        return initialized_;
    }
    
private:
    bool initialized_;
    DeviceType device_type_;
    ::sycl::queue queue_;
};

// Device类实现
Device::Device() : impl_(new Impl()) {}

Device::~Device() {}

RetCode Device::Init(DeviceType type) {
    return impl_->Init(type);
}

RetCode Device::GetDeviceProperties(DeviceProperties* prop) const {
    return impl_->GetDeviceProperties(prop);
}

::sycl::queue& Device::GetQueue() {
    return impl_->GetQueue();
}

RetCode Device::Synchronize() {
    return impl_->Synchronize();
}

DeviceType Device::GetDeviceType() const {
    return impl_->GetDeviceType();
}

bool Device::IsInitialized() const {
    return impl_->IsInitialized();
}

}}} // namespace ppl::kernel::sycl