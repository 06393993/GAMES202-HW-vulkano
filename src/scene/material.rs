// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use super::{shaders::ShadersT, Camera};

pub trait UniformT: Sized + Send + Sync + 'static {
    fn update_model_matrix(&mut self, mat: [f32; 16]);
    fn update_view_matrix(&mut self, mat: [f32; 16]);
    fn update_proj_matrix(&mut self, mat: [f32; 16]);

    fn update_view_proj_matrix_from_camera(&mut self, camera: &Camera) {
        self.update_view_matrix(camera.get_view_transform().to_array());
        self.update_proj_matrix(camera.get_projection_transform().to_array());
    }
}

pub trait Material {
    type Uniform: UniformT;
    type Shaders: ShadersT;

    fn create_uniform(&self) -> Self::Uniform;
}
