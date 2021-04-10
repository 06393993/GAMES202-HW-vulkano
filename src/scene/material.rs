// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::sync::Arc;

use vulkano::{
    command_buffer::{pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder},
    descriptor::{descriptor_set::DescriptorSet, pipeline_layout::PipelineLayoutAbstract},
    device::{Device, Queue},
};

use super::{shaders::ShadersT, Camera};
use crate::errors::*;

pub trait SetCamera {
    fn set_model_matrix(&mut self, mat: [f32; 16]);
    fn set_view_matrix(&mut self, mat: [f32; 16]);
    fn set_proj_matrix(&mut self, mat: [f32; 16]);

    fn set_view_proj_matrix_from_camera(&mut self, camera: &Camera) {
        self.set_view_matrix(camera.get_view_transform().to_array());
        self.set_proj_matrix(camera.get_projection_transform().to_array());
    }
}

pub trait UniformsT: Sized + Send + Sync + 'static {
    fn update_buffers(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
    ) -> Result<()>;
    fn create_descriptor_sets(
        &self,
        pipeline_layout: &dyn PipelineLayoutAbstract,
    ) -> Result<Vec<Arc<dyn DescriptorSet + Send + Sync + 'static>>>;
}

pub trait Material {
    type Uniforms: UniformsT;
    type Shaders: ShadersT;

    fn create_uniforms(&self, device: Arc<Device>, queue: Arc<Queue>) -> Result<Self::Uniforms>;
}
