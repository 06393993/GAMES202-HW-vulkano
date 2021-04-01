mod camera;
mod shaders;

use std::sync::Arc;

use vulkano::{
    buffer::{
        device_local::DeviceLocalBuffer, immutable::ImmutableBuffer, BufferAccess, BufferUsage,
        TypedBufferAccess,
    },
    command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents},
    descriptor::descriptor_set::{DescriptorSet, PersistentDescriptorSet},
    device::{Device, Queue},
    format::{ClearValue, Format},
    framebuffer::{Framebuffer, RenderPassAbstract, Subpass},
    image::traits::ImageViewAccess,
    pipeline::{
        viewport::{Scissor, Viewport},
        GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sync::GpuFuture,
};

struct NDCSpace;
struct ViewSpace;
struct WorldSpace;

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

vulkano::impl_vertex!(Vertex, position);

#[derive(Default, Copy, Clone)]
struct Uniform {
    color: [f32; 4],
}

pub struct State {
    pub color: [f32; 3],
}

pub struct Renderer {
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    descriptor_set: Arc<dyn DescriptorSet + Send + Sync>,
    uniform_buffer: Arc<dyn TypedBufferAccess<Content = Uniform> + Send + Sync>,
}

impl Renderer {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Renderer {
        let (vertex_buffer, vertex_buffer_init) = ImmutableBuffer::from_iter(
            vec![
                Vertex {
                    position: [-0.5, -0.5],
                },
                Vertex {
                    position: [0.0, 0.5],
                },
                Vertex {
                    position: [0.5, -0.25],
                },
            ]
            .into_iter(),
            BufferUsage::vertex_buffer(),
            queue.clone(),
        )
        .unwrap();
        vertex_buffer_init
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
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
            .unwrap(),
        );
        let vs = shaders::vertex_shader::Shader::load(device.clone())
            .expect("failed to create shader module for vertex shader");
        let fs = shaders::fragment_shader::Shader::load(device.clone())
            .expect("failed to create shader module for fragment shader");
        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
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
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        );
        let uniform_buffer = DeviceLocalBuffer::<Uniform>::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            vec![queue.family()].into_iter(),
        )
        .unwrap();
        let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(uniform_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );
        Renderer {
            render_pass,
            pipeline,
            vertex_buffer,
            descriptor_set,
            uniform_buffer,
        }
    }

    pub fn draw_commands<P>(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<impl ImageViewAccess + Send + Sync + 'static>,
        state: &State,
    ) {
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(image.clone())
                .unwrap()
                .build()
                .unwrap(),
        );
        cmd_buf_builder
            .update_buffer(
                self.uniform_buffer.clone(),
                Uniform {
                    color: [state.color[0], state.color[1], state.color[2], 1.0],
                },
            )
            .unwrap()
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::Inline,
                vec![ClearValue::None],
            )
            .unwrap()
            .draw(
                self.pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone()],
                self.descriptor_set.clone(),
                (),
            )
            .unwrap()
            .end_render_pass()
            .unwrap();
    }
}
