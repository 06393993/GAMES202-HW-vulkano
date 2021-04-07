// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use crate::impl_shaders;

pub mod texture_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/scene/shaders/phong/vertex_shader.glsl",
        define: [("WITH_TEXTURE", "1")],
    }
}

pub mod texture_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/scene/shaders/phong/fragment_shader.glsl",
        define: [("WITH_TEXTURE", "1")],
    }
}

pub mod no_texture_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/scene/shaders/phong/vertex_shader.glsl",
    }
}

pub mod no_texture_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/scene/shaders/phong/fragment_shader.glsl",
    }
}

fn __() {
    let _ = include_bytes!("fragment_shader.glsl");
    let _ = include_bytes!("vertex_shader.glsl");
}

impl_shaders!(
    TextureShaders,
    texture_vertex_shader,
    texture_fragment_shader
);
impl_shaders!(
    NoTextureShaders,
    no_texture_vertex_shader,
    no_texture_fragment_shader
);
