#version 450

layout(binding = 0) uniform UniformBufferObject {
  mat4 view;
  mat4 proj;
  vec4 color;
}
ubo;

layout(location = 0) in vec2 position;

layout(location = 0) out vec4 out_color;

void main() {
  gl_Position = ubo.proj * ubo.view * vec4(position, 0.0, 1.0);
  out_color = ubo.color;
}
