// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use super::super::material::SetCamera;
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

impl_shaders!(Shaders, vertex_shader, fragment_shader, {
    uniform: {
        layout: 0,
        ty: "buffer",
        def: {
            pub model: [f32; 16],
            pub view: [f32; 16],
            pub proj: [f32; 16],
            pub light_intensity: f32,
            pub light_color: [f32; 4],
        },
    },
});

impl SetCamera for ShadersUniforms {
    fn set_model_matrix(&mut self, mat: [f32; 16]) {
        self.uniform.model.copy_from_slice(&mat);
    }

    fn set_view_matrix(&mut self, mat: [f32; 16]) {
        self.uniform.view.copy_from_slice(&mat);
    }

    fn set_proj_matrix(&mut self, mat: [f32; 16]) {
        self.uniform.proj.copy_from_slice(&mat);
    }
}
