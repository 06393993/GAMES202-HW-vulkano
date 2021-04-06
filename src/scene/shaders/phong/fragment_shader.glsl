#version 450

layout(binding = 1) uniform UniformBufferObject {
  vec4 kd;
  vec4 ks;
  vec4 light_pos;
  vec4 camera_pos;
  float light_intensity;
}
ubo;
layout(binding = 2) uniform sampler2D tex_sampler;

layout(location = 0) in vec2 texture_coord;
layout(location = 1) in vec3 frag_pos;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec4 f_color;

void main() {
  vec3 color = pow(texture(tex_sampler, texture_coord).rgb, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 light_pos = ubo.light_pos.xyz;
  vec3 light_direction = normalize(light_pos - frag_pos);
  vec3 normal = normalize(in_normal);
  float diff = max(dot(light_direction, normal), 0.0);
  float light_atten_coff = ubo.light_intensity / length(light_pos - frag_pos);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 view_direction = normalize(ubo.camera_pos.xyz - frag_pos);
  float spec = 0.0;
  vec3 reflect_direction = reflect(-light_direction, normal);
  spec = pow(max(dot(view_direction, reflect_direction), 0.0), 35.0);
  vec3 specular = ubo.ks.xyz * light_atten_coff * spec;

  f_color = vec4(pow((ambient + diffuse + specular), vec3(1.0 / 2.2)), 1.0);
}