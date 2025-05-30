# ppl.kernel.sycl

ppl.kernel.sycl 是基于 SYCL 实现的高性能计算库，灵感来源于 ppl.kernel.cuda。它提供了一系列优化的内核函数，可以在支持 SYCL 的设备上运行，包括 Intel GPU、CPU 和其他支持 SYCL 的硬件。

## 特性

- 跨平台支持：通过 SYCL 抽象层，可以在多种设备（CPU、GPU、FPGA等）上运行相同的代码
- 与 CUDA 版本兼容的 API：保持与 ppl.kernel.cuda 相似的 API 设计，便于迁移
- 模块化设计：将功能分为算术、内存、神经网络等不同模块
- 异常处理：完善的错误处理机制，确保库的稳定性
- 性能优化：根据设备特性自动调整工作组大小等参数

## 支持的操作

- 算术运算：Add, Sub, Mul, Div
- 一元运算：Abs, Neg, Exp, Log, Sqrt
- 神经网络操作：ReLU, LeakyReLU
- 归约操作：Sum, Max, Min, Prod, Mean
- 数据重排：格式转换（NCHW <-> NHWC）、类型转换
- 内存操作：Malloc, Free, Memcpy, Memset

## 编译要求

- Intel ICPX 编译器或 Intel LLVM Clang++ 编译器
- CMake 3.14 或更高版本
- Intel oneAPI DPC++ 库

## 编译说明

使用 Intel ICPX 编译器：

```bash
# 设置 Intel oneAPI 环境
source /opt/intel/oneapi/setvars.sh

# 使用 ICPX 编译器
export CXX=icpx
# or clang++
export CXX=clang++

# 创建构建目录并编译
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
