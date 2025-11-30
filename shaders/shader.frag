#version 450

//outColor outputted from this fragmen shader (on position 0)
layout(location = 0) out vec4 outColor;
//we get the color from the vertex shader we get it from position 0, no need for it to have the same name really
layout(location = 0) in vec3 fragColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}