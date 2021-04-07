#version 450

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
}
ubo;

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;
#ifdef WITH_TEXTURE
layout(location = 2) in vec2 in_texture_coord;
#endif

#ifdef WITH_TEXTURE
layout(location = 0) out vec2 texture_coord;
#endif
layout(location = 1) out vec3 frag_pos;
layout(location = 2) out vec3 normal;

void main() {
  frag_pos = (ubo.model * vec4(in_position.xyz, 1.0)).xyz;
  normal = (ubo.model * vec4(in_normal.xyz, 0.0)).xyz;

  gl_Position = ubo.proj * ubo.view * vec4(frag_pos, 1.0);

#ifdef WITH_TEXTURE
  texture_coord = in_texture_coord;
#endif
}
