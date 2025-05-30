#ifndef PPL_KERNEL_SYCL_COMMON_DEVICE_H_
#define PPL_KERNEL_SYCL_COMMON_DEVICE_H_

#include "types.h"
#include <sycl/sycl.hpp>
#include <string>
#include <memory>

namespace ppl { namespace kernel { namespace sycl {

// 设备类型枚举
enum class DeviceType {
    CPU = 0,
    GPU = 1,
    FPGA = 2,
    ACCELERATOR = 3,
    ANY = 4
};

// 设备属性结构
struct DeviceProperties {
    std::string name;
    DeviceType type;
    size_t max_work_group_size;
    size_t max_compute_units;
    size_t global_mem_size;
    size_t local_mem_size;
    size_t max_mem_alloc_size;
    bool has_unified_memory;
};

// 设备接口类
class Device {
public:
    Device();
    ~Device();

    // 初始化设备
    RetCode Init(DeviceType type = DeviceType::ANY);
    
    // 获取设备属性
    RetCode GetDeviceProperties(DeviceProperties* prop) const;
    
    // 获取SYCL队列
    ::sycl::queue& GetQueue();
    
    // 同步设备
    RetCode Synchronize();
    
    // 获取设备类型
    DeviceType GetDeviceType() const;
    
    // 检查设备是否已初始化
    bool IsInitialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}}} // namespace ppl::kernel::sycl

#endif // PPL_KERNEL_SYCL_COMMON_DEVICE_H_