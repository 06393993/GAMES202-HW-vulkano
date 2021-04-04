pub mod light;

use std::sync::Arc;

use vulkano::{
    descriptor::pipeline_layout::PipelineLayoutDesc,
    device::Device,
    pipeline::shader::{GraphicsEntryPoint, ShaderInterfaceDef},
};

use crate::errors::*;

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

#[macro_export]
macro_rules! impl_shaders {
    ($id:ident, $vs_mod:ident, $fs_mod:ident) => {
        pub struct $id {
            vertex_shader: $vs_mod::Shader,
            fragment_shader: $fs_mod::Shader,
        }

        impl $crate::scene::shaders::ShadersT for $id {
            type VertexShaderLayout = $vs_mod::Layout;
            type VertexShaderMainInput = $vs_mod::MainInput;
            type VertexShaderMainOutput = $vs_mod::MainOutput;
            type FragmentShaderLayout = $fs_mod::Layout;
            type FragmentShaderMainInput = $fs_mod::MainInput;
            type FragmentShaderMainOutput = $fs_mod::MainOutput;

            fn load(
                device: ::std::sync::Arc<::vulkano::device::Device>,
            ) -> $crate::errors::Result<Self> {
                use $crate::errors::*;
                Ok(Self {
                    vertex_shader: $vs_mod::Shader::load(device.clone())
                        .chain_err(|| "fail to load the vertex shader")?,
                    fragment_shader: $fs_mod::Shader::load(device.clone())
                        .chain_err(|| "fail to load the fragment shader")?,
                })
            }

            fn vertex_shader_main_entry_point(
                &self,
            ) -> ::vulkano::pipeline::shader::GraphicsEntryPoint<
                (),
                Self::VertexShaderMainInput,
                Self::VertexShaderMainOutput,
                Self::VertexShaderLayout,
            > {
                self.vertex_shader.main_entry_point()
            }

            fn fragment_shader_main_entry_point(
                &self,
            ) -> ::vulkano::pipeline::shader::GraphicsEntryPoint<
                (),
                Self::FragmentShaderMainInput,
                Self::FragmentShaderMainOutput,
                Self::FragmentShaderLayout,
            > {
                self.fragment_shader.main_entry_point()
            }
        }
    };
}
