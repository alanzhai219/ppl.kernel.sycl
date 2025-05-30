cmake_minimum_required(VERSION 3.14)

project(ppl.kernel.sycl VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含自定义模块
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(deps)

# 编译器设置
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. Please use Intel ICPX or Intel LLVM Clang++.")
endif()

# 定义库版本
set(PPLKERNELSYCL_MAJOR_VERSION 1)
set(PPLKERNELSYCL_MINOR_VERSION 0)
set(PPLKERNELSYCL_PATCH_VERSION 0)
set(PPLKERNELSYCL_VERSION "${PPLKERNELSYCL_MAJOR_VERSION}.${PPLKERNELSYCL_MINOR_VERSION}.${PPLKERNELSYCL_PATCH_VERSION}")

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含目录
include_directories(include)

# 收集源文件
file(GLOB_RECURSE PPLKERNELSYCL_SRC
    "src/common/*.cpp"
    "src/arithmetic/*.cpp"
    "src/memory/*.cpp"
    "src/nn/*.cpp"
    "src/reduce/*.cpp"
    "src/reformat/*.cpp"
    "src/unary/*.cpp"
)

# 创建库
add_library(pplkernelsycl SHARED ${PPLKERNELSYCL_SRC})
target_include_directories(pplkernelsycl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 配置安装
include(GNUInstallDirs)
install(TARGETS pplkernelsycl
    EXPORT pplkernelsycl-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# 导出目标
install(EXPORT pplkernelsycl-targets
    FILE pplkernelsycl-targets.cmake
    NAMESPACE ppl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pplkernelsycl
)

# 配置文件
configure_file(pplkernelsycl-config.cmake.in pplkernelsycl-config.cmake @ONLY)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pplkernelsycl-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pplkernelsycl
)