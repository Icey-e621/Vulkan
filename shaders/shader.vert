#version 450

//colors for each vertex
vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);
//hardcoded vertexes
vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);
//for the framebuffer 0 output to a variable called fragColor (output as first variable a var of vec3)
layout(location = 0) out vec3 fragColor;

void main() {
    //included in vulkan, the position of the vertex outputted (gl_VertexIndex = index of treating vertex for that gpu core)
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    //we put a value to fragColor according to the vertex
    fragColor = colors[gl_VertexIndex];
}