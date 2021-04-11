// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use super::super::material::SetCamera;
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

pub mod with_texture {
    use super::*;

    impl_shaders!(
        Shaders,
        texture_vertex_shader,
        texture_fragment_shader,
        {
            vs_uniform: {
                layout: 0,
                ty: "buffer",
                def: {
                    model: [f32; 16],
                    view: [f32; 16],
                    proj: [f32; 16],
                },
            },
            fs_uniform: {
                layout: 1,
                ty: "buffer",
                def: {
                    pub kd: [f32; 4],
                    pub ks: [f32; 4],
                    pub light_pos: [f32; 4],
                    pub camera_pos: [f32; 4],
                    pub light_intensity: f32,
                },
            },
            texture: {
                layout: 2,
                ty: "texture",
            },
        }
    );

    impl SetCamera for ShadersUniforms {
        fn set_model_matrix(&mut self, mat: [f32; 16]) {
            self.vs_uniform.model.copy_from_slice(&mat);
        }

        fn set_view_matrix(&mut self, mat: [f32; 16]) {
            self.vs_uniform.view.copy_from_slice(&mat);
        }

        fn set_proj_matrix(&mut self, mat: [f32; 16]) {
            self.vs_uniform.proj.copy_from_slice(&mat);
        }
    }
}

pub mod no_texture {
    use super::*;

    impl_shaders!(
        Shaders,
        no_texture_vertex_shader,
        no_texture_fragment_shader,
        {
            vs_uniform: {
                layout: 0,
                ty: "buffer",
                def: {
                    pub model: [f32; 16],
                    pub view: [f32; 16],
                    pub proj: [f32; 16],
                },
            },
            fs_uniform: {
                layout: 1,
                ty: "buffer",
                def: {
                    pub kd: [f32; 4],
                    pub ks: [f32; 4],
                    pub light_pos: [f32; 4],
                    pub camera_pos: [f32; 4],
                    pub light_intensity: f32,
                },
            },
        }
    );

    impl SetCamera for ShadersUniforms {
        fn set_model_matrix(&mut self, mat: [f32; 16]) {
            self.vs_uniform.model.copy_from_slice(&mat);
        }

        fn set_view_matrix(&mut self, mat: [f32; 16]) {
            self.vs_uniform.view.copy_from_slice(&mat);
        }

        fn set_proj_matrix(&mut self, mat: [f32; 16]) {
            self.vs_uniform.proj.copy_from_slice(&mat);
        }
    }
}
