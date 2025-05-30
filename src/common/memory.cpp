#include "syclkernel/common/memory.h"

namespace ppl { namespace kernel { namespace sycl {

RetCode Malloc(Device* device, void** ptr, size_t size) {
    if (!device || !ptr || size == 0) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    try {
        auto& queue = device->GetQueue();
        *ptr = ::sycl::malloc_device(size, queue);
        
        if (*ptr == nullptr) {
            return RetCode::OUT_OF_MEMORY;
        }
        
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        *ptr = nullptr;
        return RetCode::INTERNAL_ERROR;
    }
}

RetCode Free(Device* device, void* ptr) {
    if (!device) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    if (ptr == nullptr) {
        return RetCode::SUCCESS;
    }
    
    try {
        auto& queue = device->GetQueue();
        ::sycl::free(ptr, queue);
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        return RetCode::INTERNAL_ERROR;
    }
}

RetCode MemcpyD2D(Device* device, void* dst, const void* src, size_t size) {
    if (!device || !dst || !src || size == 0) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    try {
        auto& queue = device->GetQueue();
        queue.memcpy(dst, src, size).wait();
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        return RetCode::INTERNAL_ERROR;
    }
}

RetCode MemcpyH2D(Device* device, void* dst, const void* src, size_t size) {
    if (!device || !dst || !src || size == 0) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    try {
        auto& queue = device->GetQueue();
        queue.memcpy(dst, src, size).wait();
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        return RetCode::INTERNAL_ERROR;
    }
}

RetCode MemcpyD2H(Device* device, void* dst, const void* src, size_t size) {
    if (!device || !dst || !src || size == 0) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    try {
        auto& queue = device->GetQueue();
        queue.memcpy(dst, src, size).wait();
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        return RetCode::INTERNAL_ERROR;
    }
}

RetCode Memset(Device* device, void* ptr, int value, size_t size) {
    if (!device || !ptr || size == 0) {
        return RetCode::INVALID_VALUE;
    }
    
    if (!device->IsInitialized()) {
        return RetCode::DEVICE_ERROR;
    }
    
    try {
        auto& queue = device->GetQueue();
        queue.memset(ptr, value, size).wait();
        return RetCode::SUCCESS;
        
    } catch (const ::sycl::exception& e) {
        return RetCode::INTERNAL_ERROR;
    }
}

}}} // namespace ppl::kernel::sycl