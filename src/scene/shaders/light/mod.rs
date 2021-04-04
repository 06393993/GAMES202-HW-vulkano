// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use crate::impl_shaders;

pub mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/scene/shaders/light/vertex_shader.glsl",
    }
}

pub mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/scene/shaders/light/fragment_shader.glsl",
    }
}

fn __() {
    let _ = include_bytes!("fragment_shader.glsl");
    let _ = include_bytes!("vertex_shader.glsl");
}

impl_shaders!(Shaders, vertex_shader, fragment_shader);
