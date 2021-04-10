#![recursion_limit = "1024"]

mod scene;
mod support;

use std::{
    cell::RefCell,
    path::PathBuf,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use euclid::{approxeq::ApproxEq, point3, vec2, vec3, Angle, Transform3D, Vector2D};
use imgui::*;
use vulkano::swapchain::Surface;
use winit::{
    dpi::LogicalPosition,
    event::{ElementState, MouseButton as WinitMouseButton, VirtualKeyCode},
    window::Window as WinitWindow,
};

#[macro_use]
extern crate error_chain;

use scene::{
    Camera, CameraControl, CameraDirection, ModelAndTexture, Renderer as SceneRenderer,
    State as SceneState, ViewSpace,
};

mod errors {
    error_chain! {}

    pub fn eprint_chained_err(e: &Error) {
        eprintln!("error: {}", e);

        for e in e.iter().skip(1) {
            eprintln!("caused by: {}", e);
        }

        if let Some(backtrace) = e.backtrace() {
            eprintln!("backtrace: {:?}", backtrace);
        }
    }
}

use errors::*;

fn select_model_and_texture_files() -> Result<Option<ModelAndTexture>> {
    let model_path =
        tinyfiledialogs::open_file_dialog("select model file", "", Some((&["*.obj"], "")));
    let model_path = if let Some(model_path) = model_path {
        PathBuf::from(model_path)
    } else {
        return Ok(None);
    };
    ModelAndTexture::load(&model_path).map(Some)
}

struct Application {
    surface: Arc<Surface<WinitWindow>>,
    scene_renderer: Rc<RefCell<SceneRenderer>>,

    mouse_middle_button_held: bool,
    cursor_lock_position: Option<LogicalPosition<f64>>,
    cursor_position: LogicalPosition<f64>,

    color_picker_visible: bool,
    color: [f32; 3],
    recent_frame_times: Vec<Instant>,
    camera: Option<Camera>,
    camera_speed: f32,
    model_path: Option<String>,
    start_time: Instant,
}

impl support::ApplicationT for Application {
    fn new(surface: Arc<Surface<WinitWindow>>, scene_renderer: Rc<RefCell<SceneRenderer>>) -> Self {
        Application {
            surface,
            scene_renderer,

            cursor_position: LogicalPosition::new(0.0, 0.0),
            mouse_middle_button_held: false,
            cursor_lock_position: None,

            color_picker_visible: false,
            color: [1.0, 0.0, 0.0],
            recent_frame_times: vec![],
            camera: None,
            camera_speed: 1.0,
            model_path: None,
            start_time: Instant::now(),
        }
    }

    fn get_scene_state(&mut self) -> Result<SceneState> {
        let time_elapsed = self.start_time.elapsed();
        let point_light_transform = Transform3D::identity()
            .then_scale(0.1, 0.1, 0.1)
            .then_translate(vec3(
                2.0 * (time_elapsed.as_secs_f32() * 6.0).sin(),
                3.0 * (time_elapsed.as_secs_f32() * 4.0).cos(),
                2.0 * (time_elapsed.as_secs_f32() * 2.0).cos(),
            ));
        let speed = Angle::pi() / 10.0;
        let model_transform = Transform3D::identity()
            .then_translate(vec3(0.0, -2.0, 0.0))
            .then_rotate(0.0, 1.0, 0.0, speed * time_elapsed.as_secs_f32());
        Ok(SceneState {
            point_light_transform,
            color: self.color,
            camera: self
                .get_camera_mut()
                .chain_err(|| "fail to get camera")?
                .clone(),
            model_transform,
        })
    }

    fn update_ui(&mut self, ui: &mut Ui) -> Result<()> {
        let now = Instant::now();
        self.recent_frame_times.push(now);
        self.recent_frame_times
            .retain(|frame_time| now.duration_since(*frame_time) < Duration::from_secs(1));

        self.update_camera_from_key_state(
            &ui.io().keys_down,
            Duration::from_secs_f32(ui.io().delta_time),
        )
        .chain_err(|| "fail to update the camera from key state")?;
        let [cursor_x, cursor_y] = ui.io().mouse_pos;
        self.cursor_position = LogicalPosition::new(cursor_x.into(), cursor_y.into());

        Window::new(im_str!("Hello world"))
            .size([300.0, 110.0], Condition::FirstUseEver)
            .build(ui, || {
                ui.text(format!("FPS {}", self.recent_frame_times.len()));
                if ui.small_button(im_str!("togle color picker")) {
                    self.color_picker_visible = !self.color_picker_visible;
                }
                ui.text(format!(
                    "color = ({}, {}, {})",
                    self.color[0], self.color[1], self.color[2]
                ));

                if ui.small_button(im_str!("select model files")) {
                    let res = select_model_and_texture_files()
                        .chain_err(|| "fail to load the model file or the texture file");
                    match res {
                        Ok(Some(model_and_texture)) => {
                            if let Err(ref e) = self
                                .scene_renderer
                                .borrow_mut()
                                .load_model_and_texture(model_and_texture)
                            {
                                eprint_chained_err(e);
                            }
                        }
                        Ok(None) => (), /* do nothing, the user cancel the operation */
                        Err(ref e) => eprint_chained_err(e),
                    }
                }
                if let Some(ref model_path) = self.model_path {
                    ui.text(format!("model path: {}", model_path));
                }
            });
        if self.color_picker_visible {
            let editable_color: EditableColor = (&mut self.color).into();
            let cp = ColorPicker::new(im_str!("color_picker"), editable_color);
            cp.build(&ui);
        }

        Ok(())
    }

    fn on_mouse_move(&mut self, (delta_x, delta_y): (f64, f64)) -> Result<()> {
        if let Some(location) = self.cursor_lock_position {
            self.surface
                .window()
                .set_cursor_position(location)
                .chain_err(|| "fail to set cursor position when trying to lock the cursor")?;
        }

        if !self.mouse_middle_button_held {
            return Ok(());
        }
        const ROTATION_SPEED: f32 = 0.001;
        let mut delta: Vector2D<f32, ViewSpace> =
            vec2(delta_x as f32, -delta_y as f32) * ROTATION_SPEED;
        if delta.length() > 1.0 {
            delta = delta.normalize();
        }
        self.rotate_camera_to(delta.to_point())
            .chain_err(|| "fail to rotate camera with the middle button held")?;
        Ok(())
    }

    fn on_mouse_button(&mut self, button: WinitMouseButton, state: ElementState) -> Result<()> {
        let window = self.surface.window();
        match button {
            WinitMouseButton::Middle => match state {
                ElementState::Pressed => {
                    self.mouse_middle_button_held = true;
                    self.cursor_lock_position.replace(self.cursor_position);
                    window.set_cursor_visible(false);
                    Ok(())
                }
                ElementState::Released => {
                    self.mouse_middle_button_held = false;
                    self.cursor_lock_position = None;
                    window.set_cursor_visible(true);
                    Ok(())
                }
            },
            _ => Ok(()),
        }
    }
}

impl CameraControl for Application {
    fn get_camera_mut(&mut self) -> Result<&mut Camera> {
        let inner_size = self.surface.window().inner_size();
        let aspect_ratio = (inner_size.width as f32) / (inner_size.height as f32);
        let fov = Angle::pi() / 4.0;
        let near = 1.0;
        let far = 100.0;
        let up = vec3(0.0, 1.0, 0.0);

        let camera = match self.camera.take() {
            Some(camera) if !camera.get_aspect_ratio().approx_eq(&aspect_ratio) => {
                let position = camera.get_position();
                Camera::new(
                    fov,
                    aspect_ratio,
                    near,
                    far,
                    &position,
                    &(position + camera.get_direction()),
                    &up,
                )
                .chain_err(|| "fail to re-create camera for app state when aspect ratio changes")?
            }
            Some(camera) => camera,
            None => Camera::new(
                fov,
                aspect_ratio,
                near,
                far,
                &point3(0.0, 0.0, 5.0),
                &point3(0.0, 0.0, 0.0),
                &vec3(0.0, 1.0, 0.0),
            )
            .chain_err(|| "fail to initialize camera for app state")?,
        };
        self.camera.replace(camera);
        Ok(self.camera.as_mut().unwrap())
    }

    fn get_speed(&self) -> f32 {
        self.camera_speed
    }
}

impl Application {
    fn update_camera_from_key_state(
        &mut self,
        key_state: &[bool; 512],
        elapsed: Duration,
    ) -> Result<()> {
        let keycode2direction = {
            use CameraDirection::*;
            use VirtualKeyCode::{A, D, S, W, X, Z};
            vec![
                (W, Forward),
                (S, Backward),
                (A, Left),
                (D, Right),
                (Z, Up),
                (X, Down),
            ]
        };
        for (virtual_keycode, direction) in keycode2direction {
            if key_state[virtual_keycode as usize] {
                self.move_camera(direction, elapsed).chain_err(|| {
                    format!("fail to move camera when moving towards {:?}", direction)
                })?;
            }
        }
        Ok(())
    }
}

fn main() {
    if let Err(ref e) = run() {
        eprint_chained_err(e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let system = support::init(file!())?;

    system.main_loop::<Application>();
}
