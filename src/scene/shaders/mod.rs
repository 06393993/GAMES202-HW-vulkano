use std::sync::Arc;

use vulkano::{
    device::Device,
    pipeline::shader::{GraphicsEntryPoint, GraphicsEntryPointAbstract},
};

use crate::errors::*;

pub(super) mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/scene/shaders/vertex_shader.glsl",
    }
}

pub(super) mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/scene/shaders/fragment_shader.glsl",
    }
}

fn __() {
    let _ = include_bytes!("fragment_shader.glsl");
    let _ = include_bytes!("vertex_shader.glsl");
}

pub trait ShadersT<'a>: Sized {
    type VertexShaderMainEntryPoint: GraphicsEntryPointAbstract<SpecializationConstants = ()> + 'a;
    type FragmentShaderMainEntryPoint: GraphicsEntryPointAbstract<SpecializationConstants = ()> + 'a;

    fn load(device: Arc<Device>) -> Result<Self>;
    fn vertex_shader_main_entry_point(&'a self) -> Self::VertexShaderMainEntryPoint;
    fn fragment_shader_main_entry_point(&'a self) -> Self::FragmentShaderMainEntryPoint;
}

pub(super) struct Shaders {
    vertex_shader: vertex_shader::Shader,
    fragment_shader: fragment_shader::Shader,
}

impl<'a> ShadersT<'a> for Shaders {
    type VertexShaderMainEntryPoint = GraphicsEntryPoint<
        'a,
        (),
        vertex_shader::MainInput,
        vertex_shader::MainOutput,
        vertex_shader::Layout,
    >;
    type FragmentShaderMainEntryPoint = GraphicsEntryPoint<
        'a,
        (),
        fragment_shader::MainInput,
        fragment_shader::MainOutput,
        fragment_shader::Layout,
    >;

    fn load(device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            vertex_shader: vertex_shader::Shader::load(device.clone())
                .chain_err(|| "fail to load the vertex shader")?,
            fragment_shader: fragment_shader::Shader::load(device.clone())
                .chain_err(|| "fail to load the fragment shader")?,
        })
    }

    fn vertex_shader_main_entry_point(&'a self) -> Self::VertexShaderMainEntryPoint {
        self.vertex_shader.main_entry_point()
    }

    fn fragment_shader_main_entry_point(&'a self) -> Self::FragmentShaderMainEntryPoint {
        self.fragment_shader.main_entry_point()
    }
}
