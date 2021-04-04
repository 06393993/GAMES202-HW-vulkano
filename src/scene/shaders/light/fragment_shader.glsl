#version 450

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
  float light_intensity;
  vec4 light_color;
}
ubo;

layout(location = 0) out vec4 f_color;

void main() { f_color = vec4(ubo.light_color.xyz, 1.0); }
