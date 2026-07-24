cmake_minimum_required(VERSION 2.8.12)
project(gsoa-cetsp)

set(SOURCE_FILES
        src/coords_3d.cc
        src/gsoa_3d.cc
        src/gsoa_learning.cc
        src/planar_3d.cc
        src/gsoa_ring.cc
        src/route_path_utils.cc
)

# 查找 Crow 头文件路径
find_path(CROW_INCLUDE_DIR "crow.h" /usr/local/include /usr/include)

# 如果找不到头文件，则返回错误
if(NOT CROW_INCLUDE_DIR)
   message(FATAL_ERROR "Could not find Crow header")
endif()

# 设置头文件目录
include_directories(${CROW_INCLUDE_DIR})

# 添加可执行文件
add_executable(tgsoa-cetsp src/tgsoa-cetsp.cc ${SOURCE_FILES})

target_include_directories(tgsoa-cetsp PUBLIC ${CMAKE_BINARY_DIR}/include)
# 链接其他库
target_link_libraries(tgsoa-cetsp crl-algorithm crl-gui)

# 安装规则
install(TARGETS tgsoa-cetsp DESTINATION .)
install(FILES tgsoa-cetsp.cfg DESTINATION .)
install(DIRECTORY etc DESTINATION .)
