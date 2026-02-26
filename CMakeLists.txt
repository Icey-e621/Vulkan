cmake_minimum_required(VERSION 3.15)

cmake_policy(SET CMP0091 NEW)

project(HelloTriangle)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)
find_package(tinyobjloader REQUIRED)

# # set cmake standards ONLY for my file not libraries
set(CMAKE_C_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Set static linking
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# HERE WE ARE GONNA COMPILE SHADERS
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHADERS ${SHADER_DIR}/*.vert ${SHADER_DIR}/*.frag ${SHADER_DIR}/*.glsl ${SHADER_DIR}/*.comp)
set(Vulkan_GLSLC_EXECUTABLE "glslc")

foreach(SHADER IN LISTS SHADERS)
    get_filename_component(FILENAME ${SHADER} NAME)
    add_custom_command(OUTPUT ${SHADER_DIR}/${FILENAME}.spv
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.1 ${SHADER} -o ${SHADER_DIR}/${FILENAME}.spv
        DEPENDS ${SHADER}
        COMMENT "Compiling ${FILENAME}")
    list(APPEND SPV_SHADERS ${SHADER_DIR}/${FILENAME}.spv)
endForeach()

add_custom_target(Shaders ALL DEPENDS ${SPV_SHADERS})

# HERE WE END COMPILING SHADERS
file(GLOB SOURCE_FILES "Src/*.cpp")

add_executable(Main ${SOURCE_FILES})

target_include_directories(Main PRIVATE include/)

target_sources(Main PRIVATE
    Src/main.cpp
)

target_link_libraries(Main
    PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
)

#validation layers
target_compile_definitions(Main PRIVATE $<$<CONFIG:Release>:NDEBUG>)
add_dependencies(Main Shaders)

# Copy specific resources to the build directory
file(GLOB SHADERS_COMPILED ${SHADER_DIR}/*.spv)
add_custom_command(TARGET Main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Main>/shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:Main>/textures
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS_COMPILED} $<TARGET_FILE_DIR:Main>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/textures/frutas.jpg
    ${CMAKE_CURRENT_SOURCE_DIR}/textures/aves.jpg
    $<TARGET_FILE_DIR:Main>/textures/
    COMMENT "Copying specific resources to binary directory"
)

# Custom target to package the application for distribution
add_custom_target(Package
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/SIMD-gpu
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Main> ${CMAKE_BINARY_DIR}/SIMD-gpu/

    # COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:Main>/glfw3.dll ${CMAKE_BINARY_DIR}/SIMD-gpu/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/SIMD-gpu/shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/SIMD-gpu/textures
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS_COMPILED} ${CMAKE_BINARY_DIR}/SIMD-gpu/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/textures/frutas.jpg ${CMAKE_BINARY_DIR}/SIMD-gpu/textures/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/textures/aves.jpg ${CMAKE_BINARY_DIR}/SIMD-gpu/textures/
    COMMAND ${CMAKE_COMMAND} -E tar "cf" "SIMD-gpu.zip" --format=zip ${CMAKE_BINARY_DIR}/SIMD-gpu
    COMMENT "Packaging application to ${CMAKE_BINARY_DIR}/SIMD-gpu"
    DEPENDS Main
)