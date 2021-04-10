// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

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
use winit::event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use imgui_vulkano_renderer::Renderer as UiRenderer;

use super::scene::{Renderer as SceneRenderer, State as SceneState};
use crate::errors::*;

mod clipboard {
    use clipboard::{ClipboardContext, ClipboardProvider};
    use imgui::{ClipboardBackend, ImStr, ImString};

    pub struct ClipboardSupport(ClipboardContext);

    pub fn init() -> Option<ClipboardSupport> {
        ClipboardContext::new().ok().map(ClipboardSupport)
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

pub trait ApplicationT {
    fn new(surface: Arc<Surface<Window>>, scene_renderer: Rc<RefCell<SceneRenderer>>) -> Self;
    fn get_scene_state(&mut self) -> Result<SceneState>;
    fn update_ui(&mut self, _ui: &mut Ui) -> Result<()> {
        Ok(())
    }
    fn on_mouse_move(&mut self, _delta: (f64, f64)) -> Result<()> {
        Ok(())
    }
    fn on_mouse_button(&mut self, _button: MouseButton, _state: ElementState) -> Result<()> {
        Ok(())
    }
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
    pub scene_renderer: Rc<RefCell<SceneRenderer>>,
}

pub fn init(title: &str) -> Result<System> {
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

    let scene_renderer = Rc::new(RefCell::new(
        SceneRenderer::init(
            device.clone(),
            queue.clone(),
            format,
            surface.window().inner_size().width,
            surface.window().inner_size().height,
        )
        .chain_err(|| "fail to create scene renderer")?,
    ));

    Ok(System {
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
    })
}

impl System {
    pub fn main_loop<T: ApplicationT + 'static>(self) -> ! {
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

        let mut application = T::new(surface.clone(), scene_renderer.clone());

        let res = Arc::new(Mutex::new(Ok(())));
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

                if let Err(e) = application.update_ui(&mut ui) {
                    *control_flow = ControlFlow::Exit;
                    *res.lock().unwrap() = Err(e);
                    return;
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

                let scene_state = match application
                    .get_scene_state()
                    .chain_err(|| "fail to get scene state when trying to render the scene")
                {
                    Ok(scene_state) => scene_state,
                    Err(e) => {
                        *control_flow = ControlFlow::Exit;
                        *res.lock().unwrap() = Err(e);
                        return;
                    }
                };
                if let Err(e) = scene_renderer
                    .borrow()
                    .draw_commands(
                        &mut scene_cmd_buf_builder,
                        images[image_num].clone(),
                        &scene_state,
                    )
                    .chain_err(|| "scene renderer fail to issue draw commands")
                {
                    *control_flow = ControlFlow::Exit;
                    *res.lock().unwrap() = Err(e);
                    return;
                }
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
                        eprintln!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::LoopDestroyed => {
                let exit_code = if let Err(ref e) = *res.lock().unwrap() {
                    eprint_chained_err(e);
                    1
                } else {
                    0
                };

                platform.handle_event(imgui.io_mut(), surface.window(), &event);
                ::std::process::exit(exit_code);
            }
            event => {
                let app_event_handler_res = match event {
                    Event::WindowEvent {
                        event: WindowEvent::MouseInput { state, button, .. },
                        ..
                    } => application.on_mouse_button(button, state),
                    Event::DeviceEvent {
                        event: DeviceEvent::MouseMotion { delta },
                        ..
                    } => application.on_mouse_move(delta),
                    _ => Ok(()),
                };
                if let Err(e) = app_event_handler_res {
                    *control_flow = ControlFlow::Exit;
                    *res.lock().unwrap() = Err(e);
                    return;
                }
                platform.handle_event(imgui.io_mut(), surface.window(), &event);
            }
        })
    }
}
