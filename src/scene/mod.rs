mod camera;
mod renderer;
mod shaders;

pub use camera::{Camera, CameraControl, Direction as CameraDirection};

pub struct NDCSpace;
pub struct ViewSpace;
pub struct WorldSpace;
pub struct TriangleSpace;
pub use renderer::{Renderer, State};
