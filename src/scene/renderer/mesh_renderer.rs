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
        shader::EntryPointAbstract,
        vertex::Vertex,
        viewport::{Scissor, Viewport},
        GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sync::GpuFuture,
};

use super::{Camera, ShadersT, WorldSpace};
use crate::errors::*;

pub trait Uniform: Sized + Send + Sync + 'static {
    fn update_model_matrix(&mut self, mat: [f32; 16]);
    fn update_view_matrix(&mut self, mat: [f32; 16]);
    fn update_proj_matrix(&mut self, mat: [f32; 16]);

    fn update_view_proj_matrix_from_camera(&mut self, camera: &Camera) {
        self.update_view_matrix(camera.get_view_transform().to_array());
        self.update_proj_matrix(camera.get_projection_transform().to_array());
    }
}

// M stands for model space
pub struct Mesh<V: Vertex, U: Uniform, M> {
    renderer: Arc<Renderer<V, U>>,
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    uniform_buffer: Arc<dyn TypedBufferAccess<Content = U> + Send + Sync>,
    phantom: PhantomData<M>,
}

impl<V: Vertex, U: Uniform, M> Mesh<V, U, M> {
    pub fn draw_commands<P>(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<P>,
        framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
        mut uniform: U,
        model_transform: &Transform3D<f32, M, WorldSpace>,
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
            .draw(
                self.renderer.pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone()],
                self.descriptor_set.clone(),
                (),
            )
            .chain_err(|| "fail to add the draw command to the command builder")?
            .end_render_pass()
            .chain_err(|| "fail to add the end renderpass command to the command builder")?;
        Ok(())
    }
}

pub struct Renderer<V: Vertex, U: Uniform> {
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    phantom: PhantomData<(V, U)>,
    // The only descriptor set layout for the single uniform input
    descriptor_set_0_layout: Arc<UnsafeDescriptorSetLayout>,
}

impl<V: Vertex, U: Uniform> Renderer<V, U> {
    pub fn init<'a, S>(
        device: Arc<Device>,
        shaders: &'a S,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        S: ShadersT<'a>,
        <S::VertexShaderMainEntryPoint as EntryPointAbstract>::PipelineLayout:
            Clone + Send + Sync + 'static,
        <S::FragmentShaderMainEntryPoint as EntryPointAbstract>::PipelineLayout:
            Clone + Send + Sync + 'static,
    {
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
    pub fn create_mesh<M>(self: &Arc<Self>, vertex_data: Vec<V>) -> Result<Mesh<V, U, M>> {
        let (vertex_buffer, vertex_buffer_init) = ImmutableBuffer::from_iter(
            vertex_data.into_iter(),
            BufferUsage::vertex_buffer(),
            self.queue.clone(),
        )
        .chain_err(|| "fail to create vertex buffer")?;
        vertex_buffer_init
            .then_signal_fence_and_flush()
            .chain_err(|| "fail to signal the fence and flush when initializing the vertex buffer")?
            .wait(None)
            .chain_err(|| "fail to wait for vertex buffer being initialized")?;
        let uniform_buffer = DeviceLocalBuffer::<U>::new(
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
