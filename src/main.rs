#![recursion_limit = "1024"]

mod scene_renderer;

use imgui::*;
use std::time::{Duration, Instant};
#[macro_use]
extern crate error_chain;

use scene_renderer::State as SceneState;

mod errors {
    error_chain! {}
}

use errors::*;

mod support {
    use imgui::{Context, FontConfig, FontGlyphRanges, FontSource, Ui};
    use imgui_winit_support::{HiDpiMode, WinitPlatform};

    use vulkano::command_buffer::AutoCommandBufferBuilder;
    use vulkano::device::Queue;
    use vulkano::device::{Device, DeviceExtensions};
    use vulkano::image::{ImageUsage, SwapchainImage};
    use vulkano::instance::{Instance, PhysicalDevice};
    use vulkano::swapchain;
    use vulkano::swapchain::Surface;
    use vulkano::swapchain::{
        AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
        SwapchainCreationError,
    };
    use vulkano::sync;
    use vulkano::sync::{FlushError, GpuFuture};

    use vulkano_win::VkSurfaceBuild;
    use winit::event::{Event, WindowEvent};
    use winit::event_loop::{ControlFlow, EventLoop};
    use winit::window::{Window, WindowBuilder};

    use std::sync::Arc;

    use imgui_vulkano_renderer::Renderer as UiRenderer;

    use super::scene_renderer::{Renderer as SceneRenderer, State as SceneState};

    mod clipboard {
        use clipboard::{ClipboardContext, ClipboardProvider};
        use imgui::{ClipboardBackend, ImStr, ImString};

        pub struct ClipboardSupport(ClipboardContext);

        pub fn init() -> Option<ClipboardSupport> {
            ClipboardContext::new()
                .ok()
                .map(|ctx| ClipboardSupport(ctx))
        }

        impl ClipboardBackend for ClipboardSupport {
            fn get(&mut self) -> Option<ImString> {
                self.0.get_contents().ok().map(|text| text.into())
            }
            fn set(&mut self, text: &ImStr) {
                let _ = self.0.set_contents(text.to_str().to_owned());
            }
        }
    }

    pub trait AppStateT: Default {
        fn get_scene_state(&self) -> SceneState;
    }

    pub struct System {
        pub event_loop: EventLoop<()>,
        pub device: Arc<Device>,
        pub queue: Arc<Queue>,
        pub surface: Arc<Surface<Window>>,
        pub swapchain: Arc<Swapchain<Window>>,
        pub images: Vec<Arc<SwapchainImage<Window>>>,
        pub imgui: Context,
        pub platform: WinitPlatform,
        pub ui_renderer: UiRenderer,
        pub font_size: f32,
        pub scene_renderer: SceneRenderer,
    }

    pub fn init(title: &str) -> System {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &required_extensions, None).unwrap();

        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

        let title = match title.rfind('/') {
            Some(idx) => title.split_at(idx + 1).1,
            None => title,
        };

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_title(title.to_owned())
            .build_vk_surface(&event_loop, instance.clone())
            .expect("Failed to create a window");

        let queue_family = physical
            .queue_families()
            .find(|&q| {
                // We take the first queue that supports drawing to our window.
                q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
            })
            .unwrap();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (device, mut queues) = Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (swapchain, images, format) = {
            let caps = surface.capabilities(physical).unwrap();

            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            let format = caps.supported_formats[0].0;

            let dimensions: [u32; 2] = surface.window().inner_size().into();

            let image_usage = ImageUsage {
                transfer_destination: true,
                ..ImageUsage::color_attachment()
            };

            let (swapchain, image) = Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                image_usage,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Mailbox,
                FullscreenExclusive::Default,
                true,
                ColorSpace::SrgbNonLinear,
            )
            .unwrap();
            (swapchain, image, format)
        };

        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        let clipboard_backend = clipboard::init().expect("Failed to initialize clipboard");
        imgui.set_clipboard_backend(Box::new(clipboard_backend));

        let mut platform = WinitPlatform::init(&mut imgui);
        platform.attach_window(imgui.io_mut(), &surface.window(), HiDpiMode::Rounded);

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[
            FontSource::DefaultFontData {
                config: Some(FontConfig {
                    size_pixels: font_size,
                    ..FontConfig::default()
                }),
            },
            FontSource::TtfData {
                data: include_bytes!("../resources/mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 1.75,
                    glyph_ranges: FontGlyphRanges::japanese(),
                    ..FontConfig::default()
                }),
            },
        ]);

        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        let ui_renderer = UiRenderer::init(&mut imgui, device.clone(), queue.clone(), format)
            .expect("Failed to initialize UI renderer");

        let scene_renderer = SceneRenderer::init(
            device.clone(),
            queue.clone(),
            format,
            surface.window().inner_size().width,
            surface.window().inner_size().height,
        );

        System {
            event_loop,
            device,
            queue,
            surface,
            swapchain,
            images,
            imgui,
            platform,
            ui_renderer,
            font_size,
            scene_renderer,
        }
    }

    impl System {
        pub fn main_loop<F: 'static + FnMut(&mut bool, &mut Ui, T) -> T, T: 'static + AppStateT>(
            self,
            mut run_ui: F,
        ) {
            let System {
                event_loop,
                device,
                queue,
                surface,
                mut swapchain,
                mut images,
                mut imgui,
                mut platform,
                mut ui_renderer,
                scene_renderer,
                ..
            } = self;

            let mut recreate_swapchain = false;

            let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

            let mut prev_app_state: Option<T> = Some(Default::default());

            event_loop.run(move |event, _, control_flow| match event {
                Event::NewEvents(_) => {
                    // imgui.io_mut().update_delta_time(Instant::now());
                }
                Event::MainEventsCleared => {
                    platform
                        .prepare_frame(imgui.io_mut(), &surface.window())
                        .expect("Failed to prepare frame");
                    surface.window().request_redraw();
                }
                Event::RedrawRequested(_) => {
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        // TODO: recreate scene_renderer here
                        let dimensions: [u32; 2] = surface.window().inner_size().into();
                        let (new_swapchain, new_images) =
                            match swapchain.recreate_with_dimensions(dimensions) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                        images = new_images;
                        swapchain = new_swapchain;
                        recreate_swapchain = false;
                    }

                    let mut ui = imgui.frame();

                    let mut run = true;
                    let app_state = run_ui(&mut run, &mut ui, prev_app_state.take().unwrap());
                    if !run {
                        *control_flow = ControlFlow::Exit;
                    }

                    let (image_num, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                recreate_swapchain = true;
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    platform.prepare_render(&ui, surface.window());
                    let draw_data = ui.render();

                    let mut ui_cmd_buf_builder =
                        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                            .expect("Failed to create UI command buffer");

                    ui_renderer
                        .draw_commands(
                            &mut ui_cmd_buf_builder,
                            queue.clone(),
                            images[image_num].clone(),
                            draw_data,
                        )
                        .expect("Rendering failed");

                    let ui_cmd_buf = ui_cmd_buf_builder
                        .build()
                        .expect("Failed to build UI command buffer");

                    let mut scene_cmd_buf_builder =
                        AutoCommandBufferBuilder::new(device.clone(), queue.family())
                            .expect("Failed to create scene renderer command buffer");

                    scene_cmd_buf_builder
                        .clear_color_image(images[image_num].clone(), [0.0; 4].into())
                        .unwrap();

                    scene_renderer.draw_commands(
                        &mut scene_cmd_buf_builder,
                        images[image_num].clone(),
                        &app_state.get_scene_state(),
                    );
                    let scene_cmd_buf = scene_cmd_buf_builder.build().unwrap();

                    let future = previous_frame_end
                        .take()
                        .unwrap()
                        .join(acquire_future)
                        .then_execute(queue.clone(), scene_cmd_buf)
                        .unwrap()
                        .then_execute(queue.clone(), ui_cmd_buf)
                        .unwrap()
                        .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(FlushError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(sync::now(device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("Failed to flush future: {:?}", e);
                            previous_frame_end = Some(sync::now(device.clone()).boxed());
                        }
                    }
                    prev_app_state.replace(app_state);
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                event => {
                    platform.handle_event(imgui.io_mut(), surface.window(), &event);
                }
            })
        }
    }
}

struct AppState {
    color_picker_visible: bool,
    color: [f32; 3],
    recent_frame_times: Vec<Instant>,
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            color_picker_visible: false,
            color: [1.0, 0.0, 0.0],
            recent_frame_times: vec![],
        }
    }
}

impl support::AppStateT for AppState {
    fn get_scene_state(&self) -> SceneState {
        SceneState {
            color: self.color.clone(),
        }
    }
}

fn main() {
    if let Err(ref e) = run() {
        eprintln!("error: {}", e);

        for e in e.iter().skip(1) {
            eprintln!("caused by: {}", e);
        }

        if let Some(backtrace) = e.backtrace() {
            eprintln!("backtrace: {:?}", backtrace);
        }

        ::std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let system = support::init(file!());

    system.main_loop(move |_, ui, mut app_state: AppState| {
        let now = Instant::now();
        app_state.recent_frame_times.push(now);
        app_state
            .recent_frame_times
            .retain(|frame_time| now.duration_since(*frame_time) < Duration::from_secs(1));
        Window::new(im_str!("Hello world"))
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(ui, || {
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
                ui.text(format!("FPS {}", app_state.recent_frame_times.len()));
                if ui.small_button(im_str!("togle color picker")) {
                    app_state.color_picker_visible = !app_state.color_picker_visible;
                }
                ui.text(format!(
                    "color = ({}, {}, {})",
                    app_state.color[0], app_state.color[1], app_state.color[2]
                ));
            });
        if app_state.color_picker_visible {
            let editable_color: EditableColor = (&mut app_state.color).into();
            let cp = ColorPicker::new(im_str!("color_picker"), editable_color);
            cp.build(&ui);
        }
        app_state
    });
    Ok(())
}
