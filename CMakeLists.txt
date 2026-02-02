cmake_minimum_required(VERSION 3.10)

project(HelloTriangle)

set(CMAKE_C_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# 2. Fix for MSVC (Visual Studio) specifically
# MSVC reports 199711 by default regardless of the version unless you set this flag
if(MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan REQUIRED)

# HERE WE ARE GONNA COMPILE SHADERS
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
file(GLOB SHADERS ${SHADER_DIR}/*.vert ${SHADER_DIR}/*.frag ${SHADER_DIR}/*.glsl ${SHADER_DIR}/*.comp)
set(Vulkan_GLSLC_EXECUTABLE "glslc")

foreach(SHADER IN LISTS SHADERS)
    get_filename_component(FILENAME ${SHADER} NAME)
    add_custom_command(OUTPUT ${SHADER_DIR}/${FILENAME}.spv
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${SHADER} -o ${SHADER_DIR}/${FILENAME}.spv
        DEPENDS ${SHADER}
        COMMENT "Compiling ${FILENAME}")
list(APPEND SPV_SHADERS ${SHADER_DIR}/${FILENAME}.spv)
endForeach()

add_custom_target(Shaders ALL DEPENDS ${SPV_SHADERS})
# HERE WE END COMPILING SHADERS

file(GLOB SOURCE_FILES "Src/*.cpp")

add_executable(Main ${SOURCE_FILES} ${SHADERS})
target_compile_definitions(Main PRIVATE VERT_SHADER="${VERT_SHADER}" FRAG_SHADER="${FRAG_SHADER}")

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
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/dist
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Main> ${CMAKE_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:Main>/glfw3.dll ${CMAKE_BINARY_DIR}/dist/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/dist/shaders
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/dist/textures
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADERS_COMPILED} ${CMAKE_BINARY_DIR}/dist/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/textures/frutas.jpg ${CMAKE_BINARY_DIR}/dist/textures/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/textures/aves.jpg ${CMAKE_BINARY_DIR}/dist/textures/
    COMMENT "Packaging application to ${CMAKE_BINARY_DIR}/dist"
    DEPENDS Main
)