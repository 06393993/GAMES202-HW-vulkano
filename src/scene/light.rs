// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::sync::Arc;

use vulkano::{
    buffer::{device_local::DeviceLocalBuffer, BufferUsage},
    command_buffer::AutoCommandBufferBuilder,
    device::{Device, Queue},
};

use super::{
    material::{DescriptorSetBinding, DescriptorSetBindingDesc, Material, UniformsT},
    renderer::{Mesh, MeshData, MeshRenderer, SimpleVertex},
    shaders::light::Shaders as EmissiveShaders,
};
use crate::errors::*;

#[allow(dead_code)]
#[derive(Clone)]
pub struct EmissiveUniform {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
    light_intensity: f32,
    light_color: [f32; 4],
}

pub struct EmissiveUniforms {
    uniform: EmissiveUniform,
    buffer: Arc<DeviceLocalBuffer<EmissiveUniform>>,
}

impl UniformsT for EmissiveUniforms {
    fn set_model_matrix(&mut self, mat: [f32; 16]) {
        self.uniform.model.copy_from_slice(&mat);
    }

    fn set_view_matrix(&mut self, mat: [f32; 16]) {
        self.uniform.view.copy_from_slice(&mat);
    }

    fn set_proj_matrix(&mut self, mat: [f32; 16]) {
        self.uniform.proj.copy_from_slice(&mat);
    }

    fn update_buffers<P>(&self, cmd_buf_builder: &mut AutoCommandBufferBuilder<P>) -> Result<()> {
        cmd_buf_builder
            .update_buffer(self.buffer.clone(), self.uniform.clone())
            .chain_err(|| "fail to issue update buffers commands to update emissive uniform")?;
        Ok(())
    }

    fn create_descriptor_bindings(&self) -> Vec<DescriptorSetBinding> {
        vec![DescriptorSetBinding {
            index: 0,
            desc: DescriptorSetBindingDesc::Buffer(self.buffer.clone()),
        }]
    }
}

pub struct EmissiveMaterial {
    light_intensity: f32,
    light_color: [f32; 3],
}

impl EmissiveMaterial {
    fn new(light_intensity: f32, light_color: [f32; 3]) -> Self {
        Self {
            light_intensity,
            light_color,
        }
    }
}

impl Material for EmissiveMaterial {
    type Uniforms = EmissiveUniforms;
    type Shaders = EmissiveShaders;

    fn create_uniforms(&self, device: Arc<Device>, queue: Arc<Queue>) -> Result<Self::Uniforms> {
        let buffer = DeviceLocalBuffer::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            vec![queue.family()],
        )
        .chain_err(|| {
            "fail to create device local buffer to store the uniform of the emissive material"
        })?;
        Ok(EmissiveUniforms {
            uniform: EmissiveUniform {
                model: Default::default(),
                view: Default::default(),
                proj: Default::default(),
                light_intensity: self.light_intensity,
                light_color: [
                    self.light_color[0],
                    self.light_color[1],
                    self.light_color[2],
                    1.0,
                ],
            },
            buffer,
        })
    }
}

#[derive(Default, Copy, Clone)]
pub struct PointLightVertex {
    position: [f32; 4],
}

impl SimpleVertex for PointLightVertex {
    fn create_from_position(x: f32, y: f32, z: f32) -> Self {
        PointLightVertex {
            position: [x, y, z, 1.0],
        }
    }
}

vulkano::impl_vertex!(PointLightVertex, position);

pub type PointLightRenderer = MeshRenderer<PointLightVertex, EmissiveMaterial>;

pub struct PointLight<S> {
    pub material: EmissiveMaterial,
    pub mesh: Mesh<PointLightVertex, EmissiveMaterial, S>,
}

impl<S> PointLight<S> {
    pub fn new(
        mesh_renderer: Arc<PointLightRenderer>,
        light_intensity: f32,
        light_color: [f32; 3],
    ) -> Result<Self> {
        let material = EmissiveMaterial::new(light_intensity, light_color);
        let mesh = mesh_renderer
            .create_mesh(MeshData::<PointLightVertex>::cube(), &material)
            .chain_err(|| "fail to create mesh")?;
        Ok(Self { material, mesh })
    }
}
