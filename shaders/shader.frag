#version 450

//sampler les go
layout(binding = 0) uniform sampler2D texSampler;
//outColor outputted from this fragmen shader (on position 0)
layout(location = 0) out vec4 outColor;
//we get the color from the vertex shader we get it from position 0, no need for it to have the same name really, same with texture coords
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

void main() {
    outColor = texture(texSampler, fragTexCoord);
}