pub(super) mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/scene_renderer/shaders/vertex_shader.glsl",
    }
}

pub(super) mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/scene_renderer/shaders/fragment_shader.glsl",
    }
}

fn __() {
    let _ = include_bytes!("fragment_shader.glsl");
    let _ = include_bytes!("vertex_shader.glsl");
}
