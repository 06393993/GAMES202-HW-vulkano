// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

mod mesh_renderer;

use std::sync::Arc;

use euclid::Transform3D;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    device::{Device, Queue},
    format::Format,
    image::traits::ImageViewAccess,
};

use super::{
    light::{PointLight, PointLightRenderer},
    material::{Material, UniformsT},
    Camera, TriangleSpace, WorldSpace,
};
use crate::errors::*;
pub use mesh_renderer::{Mesh, MeshData, Renderer as MeshRenderer, SimpleVertex};

pub struct State {
    pub color: [f32; 3],
    pub camera: Camera,
    pub model_transform: Transform3D<f32, TriangleSpace, WorldSpace>,
}

pub struct Renderer {
    point_light_renderer: Arc<PointLightRenderer>,
    point_light: PointLight<TriangleSpace>,
}

impl Renderer {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let point_light_renderer = Arc::new(
            MeshRenderer::init(device, queue, format, width, height)
                .chain_err(|| "fail to create mesh renderer")?,
        );
        let point_light = PointLight::new(point_light_renderer.clone(), 1.0, [1.0, 0.0, 0.0])
            .chain_err(|| "fail to create point light")?;
        Ok(Self {
            point_light_renderer,
            point_light,
        })
    }

    pub fn draw_commands<P>(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<impl ImageViewAccess + Send + Sync + 'static>,
        state: &State,
    ) -> Result<()> {
        let framebuffer = self
            .point_light_renderer
            .create_framebuffer(image)
            .chain_err(|| "fail to create framebuffer")?;
        self.point_light
            .mesh
            .draw_commands(
                cmd_buf_builder,
                framebuffer,
                &state.model_transform,
                &state.camera,
            )
            .chain_err(|| "fail to issue draw command for the mesh")?;
        Ok(())
    }
}
