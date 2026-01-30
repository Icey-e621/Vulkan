#version 450

layout(location = 0) out vec2 texCoord;

void main() {
    // Generate fullscreen quad from gl_VertexID
    // Vertices: 0: bottom-left, 1: bottom-right, 2: top-left, 3: top-right
    float x = float((gl_VertexIndex & 1) ^ ((gl_VertexIndex >> 1) & 1));
    float y = float((gl_VertexIndex >> 1) & 1);
    
    gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    texCoord = vec2(x, 1.0 - y);
}