// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::sync::Arc;

use image::DynamicImage;
use vulkano::{
    buffer::BufferAccess,
    command_buffer::AutoCommandBufferBuilder,
    device::{Device, Queue},
};

use super::{shaders::ShadersT, Camera};
use crate::errors::*;

pub enum DescriptorSetBindingDesc {
    Buffer(Arc<dyn BufferAccess + Send + Sync>),
    Image(Arc<DynamicImage>),
}

pub struct DescriptorSetBinding {
    pub index: usize,
    pub desc: DescriptorSetBindingDesc,
}

pub trait UniformsT: Sized + Send + Sync + 'static {
    fn set_model_matrix(&mut self, mat: [f32; 16]);
    fn set_view_matrix(&mut self, mat: [f32; 16]);
    fn set_proj_matrix(&mut self, mat: [f32; 16]);

    fn set_view_proj_matrix_from_camera(&mut self, camera: &Camera) {
        self.set_view_matrix(camera.get_view_transform().to_array());
        self.set_proj_matrix(camera.get_projection_transform().to_array());
    }

    fn update_buffers<P>(&self, cmd_buf_builder: &mut AutoCommandBufferBuilder<P>) -> Result<()>;
    fn create_descriptor_bindings(&self) -> Vec<DescriptorSetBinding>;
}

pub trait Material {
    type Uniforms: UniformsT;
    type Shaders: ShadersT;

    fn create_uniforms(&self, device: Arc<Device>, queue: Arc<Queue>) -> Result<Self::Uniforms>;
}
