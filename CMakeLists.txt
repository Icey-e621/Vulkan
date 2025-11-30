cmake_minimum_required(VERSION 3.10)

project(HelloTriangle)

set(CMAKE_C_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

# HERE WE ARE GONNA COMPILE SHADERS
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHADERS ${SHADER_DIR}/*.vert ${SHADER_DIR}/*.frag ${SHADER_DIR}/*.glsl)
set(Vulkan_GLSLC_EXECUTABLE "C:\\VulkanSDK\\1.4.321.1\\Bin\\glslc.exe")

foreach(SHADER IN LISTS SHADERS)
    get_filename_component(EXT ${SHADER} EXT)
    string(REPLACE "." "" NOPEXT "${EXT}")
    add_custom_command(OUTPUT ${SHADER_DIR}/${NOPEXT}.spv
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER} -o ${SHADER_DIR}/${NOPEXT}.spv
        DEPENDS ${SHADER}
        COMMENT "Compiling ${FILENAME}")
list(APPEND SPV_SHADERS ${SHADER_DIR}/${NOPEXT}.spv)
endForeach()

add_custom_target(shaders ALL DEPENDS ${SPV_SHADERS})
# HERE WE END COMPILING SHADERS

file(GLOB SOURCE_FILES "Src/*.cpp")

add_executable(Main ${SOURCE_FILES} ${SHADERS})

target_include_directories(Main PRIVATE include/)

target_sources(Main PRIVATE 
    Src/main.cpp
)

target_link_libraries(Main 
    PRIVATE 
        Vulkan::Vulkan
        glfw
        glm::glm
)
add_dependencies(Main shaders)