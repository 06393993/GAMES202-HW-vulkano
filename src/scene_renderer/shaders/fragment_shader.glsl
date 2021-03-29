#version 450

layout(binding = 0) uniform UniformBufferObject{
    vec4 color;
} ubo;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = ubo.color;
}
