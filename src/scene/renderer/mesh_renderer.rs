// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::{marker::PhantomData, sync::Arc};

use euclid::Transform3D;
use vulkano::{
    buffer::{
        device_local::DeviceLocalBuffer, immutable::ImmutableBuffer, BufferAccess, BufferUsage,
        TypedBufferAccess,
    },
    command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents},
    descriptor::descriptor_set::{
        DescriptorSet, PersistentDescriptorSet, UnsafeDescriptorSetLayout,
    },
    device::{Device, Queue},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass},
    image::traits::ImageViewAccess,
    pipeline::{
        vertex::Vertex as VertexT,
        viewport::{Scissor, Viewport},
        GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sync::GpuFuture,
};

use super::{Camera, ShadersT, WorldSpace};
use crate::errors::*;

pub trait UniformT: Sized + Send + Sync + 'static {
    fn update_model_matrix(&mut self, mat: [f32; 16]);
    fn update_view_matrix(&mut self, mat: [f32; 16]);
    fn update_proj_matrix(&mut self, mat: [f32; 16]);

    fn update_view_proj_matrix_from_camera(&mut self, camera: &Camera) {
        self.update_view_matrix(camera.get_view_transform().to_array());
        self.update_proj_matrix(camera.get_projection_transform().to_array());
    }
}

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

pub trait Material {
    type Uniform: UniformT;
    type Shaders: ShadersT;
}

// S stands for model space
pub struct Mesh<V: VertexT, M: Material, S> {
    renderer: Arc<Renderer<V, M>>,
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    index_buffer: Arc<ImmutableBuffer<[u16]>>,
    descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    uniform_buffer: Arc<dyn TypedBufferAccess<Content = M::Uniform> + Send + Sync>,
    phantom: PhantomData<S>,
}

impl<V: VertexT, M: Material, S> Mesh<V, M, S> {
    pub fn draw_commands<P>(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<P>,
        framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
        mut uniform: M::Uniform,
        model_transform: &Transform3D<f32, S, WorldSpace>,
        camera: &Camera,
    ) -> Result<()> {
        uniform.update_model_matrix(model_transform.to_array());
        uniform.update_view_proj_matrix_from_camera(camera);
        cmd_buf_builder
            .update_buffer(self.uniform_buffer.clone(), uniform)
            .chain_err(|| {
                "fail to add the update buffer for uniform command to the command builder"
            })?
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::Inline,
                vec![ClearValue::None],
            )
            .chain_err(|| "fail to add the begin renderpass command to the command builder")?
            .draw_indexed(
                self.renderer.pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone()],
                self.index_buffer.clone(),
                self.descriptor_set.clone(),
                (),
            )
            .chain_err(|| "fail to add the draw command to the command builder")?
            .end_render_pass()
            .chain_err(|| "fail to add the end renderpass command to the command builder")?;
        Ok(())
    }
}

pub struct Renderer<V: VertexT, M: Material> {
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    phantom: PhantomData<(V, M)>,
    // The only descriptor set layout for the single uniform input
    descriptor_set_0_layout: Arc<UnsafeDescriptorSetLayout>,
}

impl<V: VertexT, M: Material> Renderer<V, M> {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let shaders = M::Shaders::load(device.clone()).chain_err(|| "fail to load shaders")?;
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        load: DontCare,
                        store: Store,
                        format: format,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .chain_err(|| "fail to create render pass when initializing renderer")?,
        );
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
                .render_pass(
                    Subpass::from(render_pass.clone(), 0)
                        .expect("fail to retrieve the first subpass from the renderpass"),
                )
                .build(device.clone())
                .chain_err(|| "fail to create graphics pipeline")?,
        );
        let descriptor_set_0_layout = match pipeline.layout().descriptor_set_layout(0) {
            Some(layout) => layout.clone(),
            None => {
                return Err(
                    "can't find the first descriptor that is supposed to be bound with \
                    the uniform"
                        .into(),
                )
            }
        };
        Ok(Self {
            device,
            queue,
            render_pass,
            pipeline,
            descriptor_set_0_layout,
            phantom: PhantomData,
        })
    }

    // M is the model space
    pub fn create_mesh<S>(self: &Arc<Self>, data: MeshData<V>) -> Result<Mesh<V, M, S>> {
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
            .chain_err(|| "fail to signal the fence and flush when initializing the vertex buffer and the index buffer")?
            .wait(None)
            .chain_err(|| "fail to wait for the vertex buffer and the index buffer being initialized")?;

        let uniform_buffer = DeviceLocalBuffer::<M::Uniform>::new(
            self.device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            vec![self.queue.family()].into_iter(),
        )
        .chain_err(|| "fail to create uniform buffer")?;

        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(self.descriptor_set_0_layout.clone())
                .add_buffer(uniform_buffer.clone())
                .chain_err(|| "fail to add the uniform buffer to the descriptor set")?
                .build()
                .chain_err(|| "fail to create the descriptor set for the uniform")?,
        );
        Ok(Mesh {
            renderer: self.clone(),
            vertex_buffer,
            index_buffer,
            descriptor_set,
            uniform_buffer,
            phantom: PhantomData,
        })
    }

    pub fn create_framebuffer(
        &self,
        image: Arc<impl ImageViewAccess + Send + Sync + 'static>,
    ) -> Result<Arc<dyn FramebufferAbstract + Sync + Send>> {
        Ok(Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(image.clone())
                .chain_err(|| "fail to add the image to the framebuffer")?
                .build()
                .chain_err(|| "fail to create the framebuffer to draw on")?,
        ))
    }
}
