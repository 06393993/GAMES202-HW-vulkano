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
    shaders::{Shaders, ShadersT},
    Camera, TriangleSpace, WorldSpace,
};
use crate::errors::*;
use mesh_renderer::{Mesh, Renderer as MeshRenderer};

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

vulkano::impl_vertex!(Vertex, position);

// Uniform object may not be read from the CPU
#[allow(dead_code)]
#[derive(Default)]
struct Uniform {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
    color: [f32; 4],
}

pub struct State {
    pub color: [f32; 3],
    pub camera: Camera,
    pub model_transform: Transform3D<f32, TriangleSpace, WorldSpace>,
}

impl mesh_renderer::Uniform for Uniform {
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

pub struct Renderer {
    mesh_renderer: Arc<MeshRenderer<Vertex, Uniform>>,
    mesh: Mesh<Vertex, Uniform, TriangleSpace>,
}

impl Renderer {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let shaders = Shaders::load(device.clone()).chain_err(|| "fail to load shaders")?;
        let mesh_renderer = Arc::new(
            MeshRenderer::init(device, &shaders, queue, format, width, height)
                .chain_err(|| "fail to create mesh renderer")?,
        );
        let mesh = mesh_renderer
            .create_mesh(vec![
                Vertex {
                    position: [-0.5, -0.5],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.5, -0.25],
                },
            ])
            .chain_err(|| "fail to create mesh")?;
        Ok(Self {
            mesh_renderer,
            mesh,
        })
    }

    pub fn draw_commands<P>(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<impl ImageViewAccess + Send + Sync + 'static>,
        state: &State,
    ) -> Result<()> {
        let framebuffer = self
            .mesh_renderer
            .create_framebuffer(image)
            .chain_err(|| "fail to create framebuffer")?;
        let mut uniform = Uniform::default();
        uniform.color[0..3].copy_from_slice(&state.color);
        uniform.color[3] = 1.0;
        self.mesh
            .draw_commands(
                cmd_buf_builder,
                framebuffer,
                uniform,
                &state.model_transform,
                &state.camera,
            )
            .chain_err(|| "fail to issue draw command for the mesh")?;
        Ok(())
    }
}
