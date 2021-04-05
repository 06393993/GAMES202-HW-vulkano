#version 450

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
layout(location = 2) in vec2 in_texture_coord;

layout(location = 0) out vec2 texture_coord;
layout(location = 1) out vec3 frag_pos;
layout(location = 2) out vec3 normal;


void main() {
  frag_pos = in_position.xyz;
  normal = in_normal.xyz;

  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_position.xyz, 1.0);

  texture_coord = in_texture_coord;
}
