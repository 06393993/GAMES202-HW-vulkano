use std::sync::Arc;

use vulkano::{
    descriptor::pipeline_layout::PipelineLayoutDesc,
    device::Device,
    pipeline::shader::{GraphicsEntryPoint, ShaderInterfaceDef},
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

pub trait ShadersT: Sized {
    type VertexShaderLayout: PipelineLayoutDesc + Clone + Send + Sync + 'static;
    type VertexShaderMainInput: ShaderInterfaceDef;
    type VertexShaderMainOutput: ShaderInterfaceDef;
    type FragmentShaderLayout: PipelineLayoutDesc + Clone + Send + Sync + 'static;
    type FragmentShaderMainInput: ShaderInterfaceDef;
    type FragmentShaderMainOutput: ShaderInterfaceDef;

    fn load(device: Arc<Device>) -> Result<Self>;
    fn vertex_shader_main_entry_point(
        &self,
    ) -> GraphicsEntryPoint<
        (),
        Self::VertexShaderMainInput,
        Self::VertexShaderMainOutput,
        Self::VertexShaderLayout,
    >;
    fn fragment_shader_main_entry_point(
        &self,
    ) -> GraphicsEntryPoint<
        (),
        Self::FragmentShaderMainInput,
        Self::FragmentShaderMainOutput,
        Self::FragmentShaderLayout,
    >;
}

pub(super) struct Shaders {
    vertex_shader: vertex_shader::Shader,
    fragment_shader: fragment_shader::Shader,
}

impl ShadersT for Shaders {
    type VertexShaderLayout = vertex_shader::Layout;
    type VertexShaderMainInput = vertex_shader::MainInput;
    type VertexShaderMainOutput = vertex_shader::MainOutput;
    type FragmentShaderLayout = fragment_shader::Layout;
    type FragmentShaderMainInput = fragment_shader::MainInput;
    type FragmentShaderMainOutput = fragment_shader::MainOutput;

    fn load(device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            vertex_shader: vertex_shader::Shader::load(device.clone())
                .chain_err(|| "fail to load the vertex shader")?,
            fragment_shader: fragment_shader::Shader::load(device.clone())
                .chain_err(|| "fail to load the fragment shader")?,
        })
    }

    fn vertex_shader_main_entry_point(
        &self,
    ) -> GraphicsEntryPoint<
        (),
        Self::VertexShaderMainInput,
        Self::VertexShaderMainOutput,
        Self::VertexShaderLayout,
    > {
        self.vertex_shader.main_entry_point()
    }

    fn fragment_shader_main_entry_point(
        &self,
    ) -> GraphicsEntryPoint<
        (),
        Self::FragmentShaderMainInput,
        Self::FragmentShaderMainOutput,
        Self::FragmentShaderLayout,
    > {
        self.fragment_shader.main_entry_point()
    }
}
