pub mod light;
pub mod phong;

use std::sync::Arc;

use vulkano::{
    command_buffer::{pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder},
    descriptor::pipeline_layout::PipelineLayoutDesc,
    descriptor::{descriptor_set::DescriptorSet, pipeline_layout::PipelineLayoutAbstract},
    device::Device,
    format::R8G8B8A8Unorm,
    image::immutable::ImmutableImage,
    pipeline::shader::{GraphicsEntryPoint, ShaderInterfaceDef},
    sampler::Sampler,
};

use crate::errors::*;

pub trait UniformsT: Send + Sync + 'static {
    fn update_buffers(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
    ) -> Result<()>;
    fn create_descriptor_sets(
        &self,
        pipeline_layout: &dyn PipelineLayoutAbstract,
    ) -> Result<Vec<Arc<dyn DescriptorSet + Send + Sync + 'static>>>;
}

pub trait ShadersT: Sized {
    type VertexShaderLayout: PipelineLayoutDesc + Clone + Send + Sync + 'static;
    type VertexShaderMainInput: ShaderInterfaceDef;
    type VertexShaderMainOutput: ShaderInterfaceDef;
    type FragmentShaderLayout: PipelineLayoutDesc + Clone + Send + Sync + 'static;
    type FragmentShaderMainInput: ShaderInterfaceDef;
    type FragmentShaderMainOutput: ShaderInterfaceDef;
    type Uniforms: UniformsT;

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

#[derive(Clone)]
pub struct Texture {
    pub image: Arc<ImmutableImage<R8G8B8A8Unorm>>,
    pub sampler: Arc<Sampler>,
}

#[macro_export]
macro_rules! define_uniforms {
    ($uniforms_name:ident, {
        $($uniform_name:ident : {$($uniform_def:tt)*},)*
    }) => {
        $crate::uniform_defs_to_struct_defs!({$($uniform_name : {$($uniform_def)*},)*});
        $crate::uniform_defs_to_struct_fields_def!(
            $uniforms_name,
            {$($uniform_name : {$($uniform_def)*},)*});
        $crate::impl_uniforms!(
            $uniforms_name,
            {$($uniform_name : {$($uniform_def)*},)*});

        impl $crate::scene::shaders::UniformsT for $uniforms_name {
            $crate::impl_update_buffers!({$($uniform_name : {$($uniform_def)*},)*});
            $crate::impl_create_descriptor_sets!({$($uniform_name : {$($uniform_def)*},)*});
        }
    };
}

#[macro_export]
macro_rules! uniform_defs_to_struct_defs {
    (@ {} ()) => ();

    (@ {
        $field_name:ident : {layout: $layout:expr, ty: "buffer", def: $def:tt,},
        $($rest:tt)*
    } ()) => (
        ::paste::paste! {
            #[derive(Clone, Default)]
            pub struct [<$field_name:camel>] $def
        }
        $crate::uniform_defs_to_struct_defs!(@ { $($rest)* } ());
    );

    (@ {$field_name:ident : {layout: $layout:expr, ty: "texture",}, $($rest:tt)*} ()) => (
        $crate::uniform_defs_to_struct_defs!(@ { $($rest)* } ());
    );

    ({$($uniform_name:ident : $uniform_def:tt,)*}) => (
        $crate::uniform_defs_to_struct_defs!(@ {$($uniform_name : $uniform_def,)*} ());
    )
}

#[macro_export]
macro_rules! uniform_defs_to_struct_fields_def {
    (@ $uniforms_name:ident, {} -> ($($result:tt)*)) => (
        ::paste::paste! {
            pub struct $uniforms_name {
                $($result)*
            }
        }
    );

    (@ $uniforms_name:ident, {
        $field_name:ident : { layout: $layout:expr, ty: "buffer", def: $def:tt, },
        $($rest:tt)*
    } -> ($($result:tt)*)) => (
        $crate::uniform_defs_to_struct_fields_def!(@ $uniforms_name, { $($rest)* } -> (
            $($result)*
            pub $field_name : [<$field_name:camel>],
            [<$field_name _buffer>] :
                ::std::sync::Arc<::vulkano::buffer::device_local::DeviceLocalBuffer<[<$field_name:camel>]>>,
        ));
    );

    (@ $uniforms_name:ident, {
        $field_name:ident : {layout: $layout:expr, ty: "texture",},
        $($rest:tt)*
    } -> ($($result:tt)*)) => (
        $crate::uniform_defs_to_struct_fields_def!(@ $uniforms_name, { $($rest)* } -> (
            $($result)*
            $field_name : $crate::scene::shaders::Texture,
        ));
    );

    ($uniforms_name:ident, {$($uniform_name:ident : $uniform_def:tt,)*}) => (
        $crate::uniform_defs_to_struct_fields_def!(
            @ $uniforms_name,
            {$($uniform_name : $uniform_def,)*} -> ());
    )
}

#[macro_export]
macro_rules! impl_uniforms {
    (@ $uniforms_name:ident, $device:ident, $queue:ident, {} -> (($($new_sig:tt)*), ($($self_init:tt)*))) => (
        ::paste::paste! {
            impl $uniforms_name {
                pub fn new($($new_sig)*) -> $crate::errors::Result<Self> {
                    use $crate::errors::*;
                    Ok(Self {
                        $($self_init)*
                    })
                }
            }
        }
    );

    (@ $uniforms_name:ident, $device:ident, $queue:ident, {
        $field_name:ident : { layout: $layout:expr, ty: "buffer", def: $def:tt, },
        $($rest:tt)*
    } -> (($($new_sig:tt)*), ($($self_init:tt)*))) => (
        $crate::impl_uniforms!(@ $uniforms_name, $device, $queue, { $($rest)* } -> ((
            $($new_sig)*
            $field_name: [<$field_name:camel>],
        ), (
            $($self_init)*
            $field_name,
            [<$field_name _buffer>]: ::vulkano::buffer::device_local::DeviceLocalBuffer::new(
                $device.clone(),
                ::vulkano::buffer::BufferUsage::uniform_buffer_transfer_destination(),
                vec![$queue.family()],
            ).chain_err(|| {
                format!(
                    "fail to create device local buffer to store the {}",
                    stringify!($filed_name),
                )
            })?,
        )));
    );

    (@ $uniforms_name:ident, $device:ident, $queue:ident, {
        $field_name:ident : {layout: $layout:expr, ty: "texture",},
        $($rest:tt)*
    } -> (($($new_sig:tt)*), ($($self_init:tt)*))) => (
        $crate::impl_uniforms!(@ $uniforms_name, $device, $queue, { $($rest)* } -> ((
            $($new_sig)*
            $field_name: $crate::scene::shaders::Texture,
        ), (
            $($self_init)*
            $field_name,
        )));
    );

    ($uniforms_name:ident, {$($uniform_name:ident : $uniform_def:tt,)*}) => (
        $crate::impl_uniforms!(
            @ $uniforms_name,
            device, queue,
            {$($uniform_name : $uniform_def,)*} -> ((
                device: ::std::sync::Arc<::vulkano::device::Device>,
                queue: ::std::sync::Arc<::vulkano::device::Queue>,
            ), ()));
    )
}

#[macro_export]
macro_rules! impl_update_buffers {
    (@ $self_:ident, $cmd_buf_builder:ident, {} ()) => (
        return Ok(());
    );

    (@ $self_:ident, $cmd_buf_builder:ident, {
        $field_name:ident : {layout: $layout:expr, ty: "buffer", def: $def:tt,},
        $($rest:tt)*
    } ()) => (
        ::paste::paste! {
            let $cmd_buf_builder = $cmd_buf_builder
                .update_buffer($self_.[<$field_name _buffer>].clone(), $self_.$field_name.clone())
                .chain_err(|| {
                    concat!(
                        "fail to issue update ",
                        stringify!($field_name),
                        " buffer commands for uniforms"
                    )
                })?;
        }
        $crate::impl_update_buffers!(@ $self_, $cmd_buf_builder, { $($rest)* } ());
    );

    (@ $self_:ident, $cmd_buf_builder:ident, {
        $field_name:ident : {layout: $layout:expr, ty: "texture",},
        $($rest:tt)*
    } ()) => (
        $crate::impl_update_buffers!(@ $self_, $cmd_buf_builder, { $($rest)* } ());
    );

    ({$($uniform_name:ident : $uniform_def:tt,)*}) => (
        fn update_buffers(
            &self,
            _cmd_buf_builder:
                &mut ::vulkano::command_buffer::AutoCommandBufferBuilder<
                    ::vulkano::command_buffer::pool::standard::StandardCommandPoolBuilder>,
        ) -> $crate::errors::Result<()> {
            use $crate::errors::*;
            // the last state unwrapped to update the buffer will define an unused cmd_buf_builder,
            // hence add an underscore as the prefix
            $crate::impl_update_buffers!(@ self, _cmd_buf_builder, {$($uniform_name : $uniform_def,)*} ());
        }
    )
}

#[macro_export]
macro_rules! impl_create_descriptor_sets {
    (@ $self_:ident, $builder:ident, $current_binding:expr, {} ()) => (
        let descriptor_set = ::std::sync::Arc::new(
            $builder.build()
                .chain_err(|| "fail to create the descriptor set for the uniforms")?
        );
        return Ok(vec![descriptor_set]);
    );

    (@ $self_:ident, $builder:ident, $current_binding:expr, {
        $field_name:ident : {layout: $layout:expr, ty: "buffer", def: $def:tt,},
        $($rest:tt)*
    } ()) => (
        ::games202_hw_vulkano_macros::add_empty_descriptor_bindings!($builder, $current_binding, $layout);
        ::paste::paste! {
            let $builder = $builder
                .add_buffer($self_.[<$field_name _buffer>].clone())
                .chain_err(|| {
                    format!(
                        "fail to add the uniform buffer to the descriptor set for the uniforms, \
                        binding = {}",
                        $layout,
                    )
                })?;
        }
        $crate::impl_create_descriptor_sets!(@ $self_, $builder, $layout, { $($rest)* } ());
    );

    (@ $self_:ident, $builder:ident, $current_binding:expr, {
        $field_name:ident : {layout: $layout:expr, ty: "texture",}, $($rest:tt)*
    } ()) => (
        ::games202_hw_vulkano_macros::add_empty_descriptor_bindings!(
            $builder,
            $current_binding,
            $layout
        );
        let $builder = $builder
            .add_sampled_image(
                $self_.$field_name.image.clone(),
                $self_.$field_name.sampler.clone()
            )
            .chain_err(|| {
                format!(
                    "fail to add the image with the sampler to the descriptor set for the \
                    uniforms,  binding = {}",
                    $layout,
                )
            })?;
        $crate::impl_create_descriptor_sets!(@ $self_, $builder, $layout, { $($rest)* } ());
    );

    ({$($uniform_name:ident : $uniform_def:tt,)*}) => (
        fn create_descriptor_sets(
            &self,
            pipeline_layout: &dyn ::vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract,
        ) -> $crate::errors::Result<::std::vec::Vec<::std::sync::Arc<
                dyn ::vulkano::descriptor::descriptor_set::DescriptorSet + ::std::marker::Send
                    + std::marker::Sync + 'static
        >>> {
                use $crate::errors::*;
            let layout = pipeline_layout
                .descriptor_set_layout(0)
                .ok_or_else(|| -> $crate::errors::Error {
                    "can't find the descriptor set at the index 0".into()
                })?;
            let descriptor_set_builder =
                ::vulkano::descriptor::descriptor_set::PersistentDescriptorSet::start(
                    layout.clone());
            $crate::impl_create_descriptor_sets!(
                @ self,
                descriptor_set_builder,
                -1,
                {$($uniform_name : $uniform_def,)*} ()
            );
        }
    )
}

#[macro_export]
macro_rules! impl_shaders {
    ($id:ident, $vs_mod:ident, $fs_mod:ident, $uniforms_def:tt) => {
        ::paste::paste! {
            $crate::define_uniforms!([<$id Uniforms>], $uniforms_def);
        }

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
            type Uniforms = ::paste::paste! { [<$id Uniforms>] };

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
