#ifndef PPL_KERNEL_SYCL_COMMON_MEMORY_H_
#define PPL_KERNEL_SYCL_COMMON_MEMORY_H_

#include "device.h"
#include "types.h"
#include <cstddef>

namespace ppl { namespace kernel { namespace sycl {

// 内存分配函数
RetCode Malloc(Device* device, void** ptr, size_t size);

// 内存释放函数
RetCode Free(Device* device, void* ptr);

// 内存拷贝函数 (设备到设备)
RetCode MemcpyD2D(Device* device, void* dst, const void* src, size_t size);

// 内存拷贝函数 (主机到设备)
RetCode MemcpyH2D(Device* device, void* dst, const void* src, size_t size);

// 内存拷贝函数 (设备到主机)
RetCode MemcpyD2H(Device* device, void* dst, const void* src, size_t size);

// 内存设置函数
RetCode Memset(Device* device, void* ptr, int value, size_t size);

}}} // namespace ppl::kernel::sycl

#endif // PPL_KERNEL_SYCL_COMMON_MEMORY_H_