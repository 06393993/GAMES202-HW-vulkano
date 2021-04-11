// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::{marker::PhantomData, sync::Arc};

use vulkano::{
    buffer::{immutable::ImmutableBuffer, BufferAccess, BufferUsage},
    command_buffer::{
        pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder, DynamicState,
    },
    descriptor::{
        descriptor_set::DescriptorSet,
        pipeline_layout::{PipelineLayout, PipelineLayoutAbstract},
    },
    device::{Device, Queue},
    framebuffer::{RenderPassAbstract, Subpass},
    pipeline::{
        depth_stencil::DepthStencil,
        vertex::Vertex as VertexT,
        viewport::{Scissor, Viewport},
        GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sync::GpuFuture,
};

use super::{
    super::shaders::{ShadersT, UniformsT},
    Material, SetCamera,
};
use crate::errors::*;

pub trait SimpleVertex: VertexT {
    fn create_from_position(x: f32, y: f32, z: f32) -> Self;
}

pub struct MeshData<V: VertexT> {
    vertices: Vec<V>,
    indices: Vec<u16>,
}

impl<V: VertexT> MeshData<V> {
    pub fn create(vertices: Vec<V>, indices: Vec<u16>) -> Result<Self> {
        for index in indices.iter() {
            if *index as usize >= vertices.len() {
                return Err(format!(
                    "index({}) exceeds the length of the vertex buffer({})",
                    index,
                    vertices.len(),
                )
                .into());
            }
        }
        Ok(Self { vertices, indices })
    }
}

impl<V: SimpleVertex> MeshData<V> {
    pub fn cube() -> Self {
        let v = V::create_from_position;
        Self::create(
            vec![
                // Front face
                v(-1.0, -1.0, 1.0),
                v(1.0, -1.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(-1.0, 1.0, 1.0),
                // Back face
                v(-1.0, -1.0, -1.0),
                v(-1.0, 1.0, -1.0),
                v(1.0, 1.0, -1.0),
                v(1.0, -1.0, -1.0),
                // Top face
                v(-1.0, 1.0, -1.0),
                v(-1.0, 1.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(1.0, 1.0, -1.0),
                // Bottom face
                v(-1.0, -1.0, -1.0),
                v(1.0, -1.0, -1.0),
                v(1.0, -1.0, 1.0),
                v(-1.0, -1.0, 1.0),
                // Right face
                v(1.0, -1.0, -1.0),
                v(1.0, 1.0, -1.0),
                v(1.0, 1.0, 1.0),
                v(1.0, -1.0, 1.0),
                // Left face
                v(-1.0, -1.0, -1.0),
                v(-1.0, -1.0, 1.0),
                v(-1.0, 1.0, 1.0),
                v(-1.0, 1.0, -1.0),
            ],
            vec![
                0, 1, 2, 0, 2, 3, // front
                4, 5, 6, 4, 6, 7, // back
                8, 9, 10, 8, 10, 11, // top
                12, 13, 14, 12, 14, 15, // bottom
                16, 17, 18, 16, 18, 19, // right
                20, 21, 22, 20, 22, 23, // left
            ],
        )
        .expect("fail to create cube")
    }
}

pub trait MeshT<S> {
    fn draw_commands(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
    ) -> Result<()>;
}

// S stands for model space
pub struct Mesh<V: VertexT, M: Material, S> {
    renderer: Arc<Renderer<V, M>>,
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    index_buffer: Arc<ImmutableBuffer<[u16]>>,
    descriptor_sets: Vec<Arc<dyn DescriptorSet + Send + Sync>>,
    phantom: PhantomData<S>,
}

impl<V: VertexT, M: Material, S> MeshT<S> for Mesh<V, M, S>
where
    <<M as Material>::Shaders as ShadersT>::Uniforms: SetCamera,
{
    fn draw_commands(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
    ) -> Result<()> {
        cmd_buf_builder
            .draw_indexed(
                self.renderer.pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone()],
                self.index_buffer.clone(),
                self.descriptor_sets.to_vec(),
                (),
            )
            .chain_err(|| "fail to add the draw command to the command builder")?;
        Ok(())
    }
}

pub struct Renderer<V: VertexT, M: Material> {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    pipeline_layout: Box<dyn PipelineLayoutAbstract>,
    phantom: PhantomData<(V, M)>,
}

type Uniforms<M> = <<M as Material>::Shaders as ShadersT>::Uniforms;

impl<V: VertexT, M: Material> Renderer<V, M> {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass<impl RenderPassAbstract + Send + Sync + 'static>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let shaders = M::Shaders::load(device.clone()).chain_err(|| "fail to load shaders")?;
        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<V>()
                .vertex_shader(shaders.vertex_shader_main_entry_point(), ())
                .viewports_scissors(
                    vec![(
                        Viewport {
                            origin: [0.0, 0.0],
                            dimensions: [width as f32, height as f32],
                            depth_range: 0.0..1.0,
                        },
                        Scissor {
                            origin: [0, 0],
                            dimensions: [width, height],
                        },
                    )]
                    .into_iter(),
                )
                .fragment_shader(shaders.fragment_shader_main_entry_point(), ())
                .depth_stencil(DepthStencil::simple_depth_test())
                .depth_write(true)
                .render_pass(subpass)
                .build(device.clone())
                .chain_err(|| "fail to create graphics pipeline")?,
        );
        let pipeline_layout = Box::new(
            PipelineLayout::new(device.clone(), pipeline.clone())
                .chain_err(|| "fail to create pipeline layout from the graphics pipeline")?,
        );
        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_layout,
            phantom: PhantomData,
        })
    }

    // M is the model space
    pub fn create_mesh<S>(
        self: &Arc<Self>,
        data: MeshData<V>,
        material: &M,
    ) -> Result<(Mesh<V, M, S>, Uniforms<M>)> {
        let MeshData {
            vertices: vertex_data,
            indices: index_data,
        } = data;
        let (vertex_buffer, vertex_buffer_init) = ImmutableBuffer::from_iter(
            vertex_data.into_iter(),
            BufferUsage::vertex_buffer(),
            self.queue.clone(),
        )
        .chain_err(|| "fail to create vertex buffer")?;
        let (index_buffer, index_buffer_init) = ImmutableBuffer::from_iter(
            index_data.into_iter(),
            BufferUsage::index_buffer(),
            self.queue.clone(),
        )
        .chain_err(|| "fail to create index buffer")?;
        vertex_buffer_init
            .join(index_buffer_init)
            .then_signal_fence_and_flush()
            .chain_err(|| {
                "fail to signal the fence and flush when initializing the vertex buffer and the \
                index buffer"
            })?
            .wait(None)
            .chain_err(|| {
                "fail to wait for the vertex buffer and the index buffer being initialized"
            })?;

        let uniforms = material
            .create_uniforms(self.device.clone(), self.queue.clone())
            .chain_err(|| "fail to create uniforms")?;
        let descriptor_sets = uniforms
            .create_descriptor_sets(self.pipeline_layout.as_ref())
            .chain_err(|| "fail to create descriptor sets for uniforms")?;
        Ok((
            Mesh {
                renderer: self.clone(),
                vertex_buffer,
                index_buffer,
                descriptor_sets,
                phantom: PhantomData,
            },
            uniforms,
        ))
    }

    pub fn get_device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn get_queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }
}
