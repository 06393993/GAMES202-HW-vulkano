// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::sync::Arc;

use euclid::{Point3D, Transform3D};
use vulkano::{
    command_buffer::{pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder},
    device::{Device, Queue},
};

use super::{
    material::{Material, SetCamera},
    renderer::{Mesh, MeshData, MeshRenderer, SimpleVertex},
    shaders::{
        light::{Shaders as EmissiveShaders, Uniform as EmissiveUniform},
        ShadersT, UniformsT,
    },
    Camera, WorldSpace,
};
use crate::errors::*;

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

type EmissiveUniforms = <EmissiveShaders as ShadersT>::Uniforms;

impl Material for EmissiveMaterial {
    type Shaders = EmissiveShaders;

    fn create_uniforms(&self, device: Arc<Device>, queue: Arc<Queue>) -> Result<EmissiveUniforms> {
        EmissiveUniforms::new(
            device,
            queue,
            EmissiveUniform {
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
        )
    }
}

#[derive(Default, Copy, Clone)]
pub struct PointLightVertex {
    position: [f32; 4],
}

vulkano::impl_vertex!(PointLightVertex, position);

impl SimpleVertex for PointLightVertex {
    fn create_from_position(x: f32, y: f32, z: f32) -> Self {
        PointLightVertex {
            position: [x, y, z, 1.0],
        }
    }
}

pub type PointLightRenderer = MeshRenderer<PointLightVertex, EmissiveMaterial>;

pub struct PointLight<S> {
    pub material: EmissiveMaterial,
    pub mesh: Mesh<PointLightVertex, EmissiveMaterial, S>,
    uniforms: EmissiveUniforms,
}

impl<S> PointLight<S> {
    pub fn new(
        mesh_renderer: Arc<PointLightRenderer>,
        light_intensity: f32,
        light_color: [f32; 3],
    ) -> Result<Self> {
        let material = EmissiveMaterial::new(light_intensity, light_color);
        let (mesh, uniforms) = mesh_renderer
            .create_mesh(MeshData::<PointLightVertex>::cube(), &material)
            .chain_err(|| "fail to create mesh")?;
        Ok(Self {
            material,
            mesh,
            uniforms,
        })
    }

    pub fn prepare_draw_commands(
        &mut self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
        model_transform: &Transform3D<f32, S, WorldSpace>,
        camera: &Camera,
    ) -> Result<()> {
        self.uniforms.set_model_matrix(model_transform.to_array());
        self.uniforms.set_view_proj_matrix_from_camera(camera);
        self.uniforms
            .update_buffers(cmd_buf_builder)
            .chain_err(|| {
                "fail to add the update buffer for uniforms command to the command builder"
            })?;
        Ok(())
    }

    pub fn get_position(&self) -> Result<Point3D<f32, WorldSpace>> {
        Transform3D::from_array(self.uniforms.uniform.model)
            .transform_point3d(Point3D::<f32, S>::origin())
            .ok_or_else(|| "invalid point light model transform".into())
    }

    pub fn get_intensity(&self) -> f32 {
        self.uniforms.uniform.light_intensity
    }
}
