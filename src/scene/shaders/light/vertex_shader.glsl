#version 450

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
  float light_intensity;
  vec4 light_color;
}
ubo;

layout(location = 0) in vec4 position;

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position.xyz, 1.0);
}
