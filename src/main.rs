#![recursion_limit = "1024"]

mod scene;
mod support;

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use euclid::{approxeq::ApproxEq, point3, vec3, Angle, Point3D, Scale, Transform3D};
use image::{io::Reader as ImageReader, DynamicImage};
use imgui::*;
use obj::{Obj, ObjData};
use winit::event::VirtualKeyCode;
#[macro_use]
extern crate error_chain;

use scene::{
    Camera, CameraControl, CameraDirection, State as SceneState, TriangleSpace, WorldSpace,
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

struct ModelAndTexture {
    obj: ObjData,
    texture: DynamicImage,
}

impl ModelAndTexture {
    fn load(obj_path: &PathBuf, texture_path: &PathBuf) -> Result<Self> {
        let obj = Obj::load(obj_path.as_path()).chain_err(|| "fail to load obj file")?;
        let texture = ImageReader::open(texture_path.as_path())
            .chain_err(|| format!("fail to open image file: {}", texture_path.display()))?
            .decode()
            .chain_err(|| "fail to decode the image")?;
        Ok(Self {
            obj: obj.data,
            texture,
        })
    }
}

fn select_model_and_texture_files() -> Result<Option<ModelAndTexture>> {
    let model_path =
        tinyfiledialogs::open_file_dialog("select model file", "", Some((&["*.obj"], "")));
    let model_path = if let Some(model_path) = model_path {
        PathBuf::from(model_path)
    } else {
        return Ok(None);
    };
    let texture_path =
        tinyfiledialogs::open_file_dialog("select texture file", "", Some((&["*.png"], "")));
    let texture_path = if let Some(texture_path) = texture_path {
        PathBuf::from(texture_path)
    } else {
        return Ok(None);
    };
    ModelAndTexture::load(&model_path, &texture_path).map(Some)
}

struct AppState {
    color_picker_visible: bool,
    color: [f32; 3],
    recent_frame_times: Vec<Instant>,
    camera: Option<Camera>,
    camera_speed: f32,
    triangle_transform: Transform3D<f32, TriangleSpace, WorldSpace>,
    model_path: Option<String>,
}

impl Default for AppState {
    fn default() -> Self {
        AppState {
            color_picker_visible: false,
            color: [1.0, 0.0, 0.0],
            recent_frame_times: vec![],
            camera: None,
            camera_speed: 0.05,
            triangle_transform: Transform3D::from_scale(Scale::new(0.5)),
            model_path: None,
        }
    }
}

impl support::AppStateT for AppState {
    fn get_scene_state(&self) -> SceneState {
        SceneState {
            color: self.color.clone(),
            camera: self.camera.as_ref().unwrap().clone(),
            model_transform: self.triangle_transform,
        }
    }
}

impl CameraControl for AppState {
    fn get_camera_mut(&mut self) -> &mut Camera {
        self.camera.as_mut().unwrap()
    }

    fn get_speed(&self) -> f32 {
        self.camera_speed
    }
}

impl AppState {
    fn update_camera(&mut self, ui: &mut imgui::Ui) -> Result<()> {
        let aspect_ratio = (ui.io().display_size[0] as f32) / (ui.io().display_size[1] as f32);
        let fov = Angle::pi() / 4.0;
        let near = 0.1;
        let far = 10.0;
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
                &point3(2.0, 2.0, 2.0),
                &Point3D::origin(),
                &vec3(0.0, 1.0, 0.0),
            )
            .chain_err(|| "fail to initialize camera for app state")?,
        };
        self.camera.replace(camera);
        let key2direction = {
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
        for (key, direction) in key2direction.into_iter() {
            if ui.io().keys_down[key as usize] {
                self.move_camera(direction, Duration::from_secs_f32(ui.io().delta_time));
            }
        }
        Ok(())
    }

    fn update_scene(&mut self, delta_time: Duration) {
        let speed = Angle::pi() / 20.0;
        self.triangle_transform =
            self.triangle_transform
                .then_rotate(0.0, 1.0, 0.0, speed * delta_time.as_secs_f32());
    }

    fn update_ui(&mut self, ui: &mut Ui) {
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
                    if let Err(ref e) = select_model_and_texture_files()
                        .chain_err(|| "fail to load the model file or the texture file")
                    {
                        eprint_chained_err(e);
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

    system.main_loop(move |_, ui, mut app_state: AppState| {
        let now = Instant::now();
        app_state.recent_frame_times.push(now);
        app_state
            .recent_frame_times
            .retain(|frame_time| now.duration_since(*frame_time) < Duration::from_secs(1));
        app_state.update_ui(ui);
        app_state
            .update_camera(ui)
            .chain_err(|| "fail to update camera in main loop")?;
        app_state.update_scene(Duration::from_secs_f32(ui.io().delta_time));
        Ok(app_state)
    });
}
