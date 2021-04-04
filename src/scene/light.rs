// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::sync::Arc;

use super::{
    material::{Material, UniformT},
    renderer::{Mesh, MeshData, MeshRenderer, SimpleVertex},
    shaders::light::Shaders as EmissiveShaders,
};
use crate::errors::*;

#[allow(dead_code)]
pub struct EmissiveUniform {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
    light_intensity: f32,
    light_color: [f32; 4],
}

impl UniformT for EmissiveUniform {
    fn update_model_matrix(&mut self, mat: [f32; 16]) {
        self.model.copy_from_slice(&mat);
    }

    fn update_view_matrix(&mut self, mat: [f32; 16]) {
        self.view.copy_from_slice(&mat);
    }

    fn update_proj_matrix(&mut self, mat: [f32; 16]) {
        self.proj.copy_from_slice(&mat);
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
    type Uniform = EmissiveUniform;
    type Shaders = EmissiveShaders;

    fn create_uniform(&self) -> Self::Uniform {
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
        }
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
        let mesh = mesh_renderer
            .create_mesh(MeshData::<PointLightVertex>::cube())
            .chain_err(|| "fail to create mesh")?;
        let material = EmissiveMaterial::new(light_intensity, light_color);
        Ok(Self { material, mesh })
    }
}
