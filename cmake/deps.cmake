# 查找SYCL依赖
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # 使用 check_cxx_source_compiles 测试 SYCL 支持
    include(CheckCXXSourceCompiles)

    # 设置一个简单的 SYCL 测试代码
    set(SYCL_TEST_CODE "
        #include <sycl/sycl.hpp>
        int main() {
            sycl::queue q;
            return 0;
        }
    ")

    # 保存当前的编译器标志
    set(CMAKE_REQUIRED_FLAGS "-fsycl")

    # 检查是否可以编译 SYCL 代码
    check_cxx_source_compiles("${SYCL_TEST_CODE}" HAVE_SYCL_SUPPORT)

    if(HAVE_SYCL_SUPPORT)
        message(STATUS "${CMAKE_CXX_COMPILER_ID} supports SYCL")
    else()
        message(WARNING "${CMAKE_CXX_COMPILER_ID} does NOT support SYCL")
    endif()

    unset(CMAKE_REQUIRED_FLAGS)
else()
    message(WARNING "Compiler is not Clang: ${CMAKE_CXX_COMPILER_ID}")
endif()

# 检查SYCL头文件是否可用
include(CheckIncludeFileCXX)
set(CMAKE_REQUIRED_FLAGS "-fsycl")
CHECK_INCLUDE_FILE_CXX("sycl/sycl.hpp" HAVE_SYCL_HPP)
if(NOT HAVE_SYCL_HPP)
    message(FATAL_ERROR "SYCL headers not found. Please ensure Intel oneAPI DPC++ Compiler is properly installed.")
endif()
