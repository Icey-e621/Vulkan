#version 450

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

// Define the blurred output image as a combined image sampler
layout(binding = 1) uniform sampler2D blurredImage;

void main() {
    // Sample from the blurred output image using texture coordinates
    outColor = texture(blurredImage, texCoord);
}