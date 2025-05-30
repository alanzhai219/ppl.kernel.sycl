#include <syclkernel/common/device.h>
#include <syclkernel/arithmetic/arithmetic.h>
#include <iostream>

int main() {
    // 初始化设备
    ppl::kernel::sycl::Device device;
    auto rc = device.Init(ppl::kernel::sycl::DeviceType::GPU);
    if (rc != ppl::kernel::sycl::RetCode::SUCCESS) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }
    
    // 打印设备信息
    ppl::kernel::sycl::DeviceProperties prop;
    device.GetDeviceProperties(&prop);
    std::cout << "Device: " << prop.name << std::endl;
    
    // 创建输入和输出数据
    const int size = 1024;
    float* input_a = nullptr;
    float* input_b = nullptr;
    float* output = nullptr;
    
    // 分配设备内存
    ppl::kernel::sycl::Malloc(&device, (void**)&input_a, size * sizeof(float));
    ppl::kernel::sycl::Malloc(&device, (void**)&input_b, size * sizeof(float));
    ppl::kernel::sycl::Malloc(&device, (void**)&output, size * sizeof(float));
    
    // 准备输入数据
    float host_input_a[size];
    float host_input_b[size];
    for (int i = 0; i < size; ++i) {
        host_input_a[i] = static_cast<float>(i);
        host_input_b[i] = static_cast<float>(i * 2);
    }
    
    // 拷贝数据到设备
    ppl::kernel::sycl::MemcpyH2D(&device, input_a, host_input_a, size * sizeof(float));
    ppl::kernel::sycl::MemcpyH2D(&device, input_b, host_input_b, size * sizeof(float));
    
    // 设置张量形状
    ppl::kernel::sycl::TensorShape shape;
    shape.dim_count = 1;
    shape.dims[0] = size;
    
    // 执行加法运算
    rc = ppl::kernel::sycl::arithmetic::Add(&device, input_a, input_b, output, shape);
    if (rc != ppl::kernel::sycl::RetCode::SUCCESS) {
        std::cerr << "Failed to execute Add operation" << std::endl;
        return -1;
    }
    
    // 拷贝结果回主机
    float host_output[size];
    ppl::kernel::sycl::MemcpyD2H(&device, host_output, output, size * sizeof(float));
    
    // 验证结果
    for (int i = 0; i < 10; ++i) {
        std::cout << host_input_a[i] << " + " << host_input_b[i] << " = " << host_output[i] << std::endl;
    }
    
    // 释放内存
    ppl::kernel::sycl::Free(&device, input_a);
    ppl::kernel::sycl::Free(&device, input_b);
    ppl::kernel::sycl::Free(&device, output);
    
    return 0;
}