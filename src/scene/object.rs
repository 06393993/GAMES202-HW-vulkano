// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::{collections::HashMap, convert::TryInto, sync::Arc};

use euclid::Point3D;
use image::RgbaImage;
use obj::{Group, IndexTuple};
use ordered_float::OrderedFloat;
use vulkano::{
    buffer::{device_local::DeviceLocalBuffer, BufferUsage},
    command_buffer::AutoCommandBufferBuilder,
    descriptor::{
        descriptor_set::{DescriptorSet, PersistentDescriptorSet},
        pipeline_layout::PipelineLayoutAbstract,
    },
    device::{Device, Queue},
    format::R8G8B8A8Unorm,
    image::{immutable::ImmutableImage, Dimensions, MipmapsCount},
    sampler::Sampler,
    sync::GpuFuture,
};

use super::{
    material::{Material, UniformsT},
    renderer::{Mesh, MeshData, MeshRenderer},
    shaders::phong::Shaders as PhongShaders,
    Camera, WorldSpace,
};
use crate::errors::*;

#[derive(Default, Copy, Clone)]
pub struct ObjectVertex {
    in_position: [f32; 4],
    in_normal: [f32; 4],
    in_texture_coord: [f32; 2],
}

vulkano::impl_vertex!(ObjectVertex, in_position, in_normal, in_texture_coord);

#[derive(Clone, Default)]
pub struct VSUniform {
    model: [f32; 16],
    view: [f32; 16],
    proj: [f32; 16],
}

#[derive(Clone)]
pub struct FSUniform {
    kd: [f32; 4],
    ks: [f32; 4],
    light_pos: [f32; 4],
    camera_pos: [f32; 4],
    light_intensity: f32,
}

pub struct ObjectUniforms {
    vs_uniform: VSUniform,
    fs_uniform: FSUniform,

    vs_uniform_buffer: Arc<DeviceLocalBuffer<VSUniform>>,
    fs_uniform_buffer: Arc<DeviceLocalBuffer<FSUniform>>,
    texture: Arc<ImmutableImage<R8G8B8A8Unorm>>,
    sampler: Arc<Sampler>,
}

impl UniformsT for ObjectUniforms {
    fn set_model_matrix(&mut self, mat: [f32; 16]) {
        self.vs_uniform.model.copy_from_slice(&mat);
    }

    fn set_view_matrix(&mut self, mat: [f32; 16]) {
        self.vs_uniform.view.copy_from_slice(&mat);
    }

    fn set_proj_matrix(&mut self, mat: [f32; 16]) {
        self.vs_uniform.proj.copy_from_slice(&mat);
    }

    fn update_buffers<P>(&self, cmd_buf_builder: &mut AutoCommandBufferBuilder<P>) -> Result<()> {
        cmd_buf_builder
            .update_buffer(self.vs_uniform_buffer.clone(), self.vs_uniform.clone())
            .chain_err(|| {
                "fail to issue update vertex shader uniform buffer commands for object uniforms"
            })?
            .update_buffer(self.fs_uniform_buffer.clone(), self.fs_uniform.clone())
            .chain_err(|| {
                "fail to issue update fragment shader uniform buffer commands for object uniforms"
            })?;
        Ok(())
    }

    fn create_descriptor_sets(
        &self,
        pipeline_layout: &dyn PipelineLayoutAbstract,
    ) -> Result<Vec<Arc<dyn DescriptorSet + Send + Sync + 'static>>> {
        let layout = pipeline_layout
            .descriptor_set_layout(0)
            .ok_or::<Error>("can't find the descriptor set at the index 0".into())?;
        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(layout.clone())
                .add_buffer(self.vs_uniform_buffer.clone())
                .chain_err(|| "fail to add the vertex shader uniform buffer to the descriptor set for the object uniforms, binding = 0")?
                .add_buffer(self.fs_uniform_buffer.clone())
                .chain_err(|| "fail to add the fragment shader uniform buffer to the descriptor set for the object uniforms, binding = 1")?
                .add_sampled_image(self.texture.clone(), self.sampler.clone())
                .chain_err(|| "fail to add the image with the sampler to the descriptor set for the object uniforms, binding = 2")?
                .build()
                .chain_err(|| "fail to create the descriptor set for the object uniforms")?
        );
        Ok(vec![descriptor_set])
    }
}

impl ObjectUniforms {
    pub fn set_light_pos(&mut self, pos: Point3D<f32, WorldSpace>) {
        self.fs_uniform.light_pos = [pos.x, pos.y, pos.z, 1.0];
    }

    pub fn set_camera_pos(&mut self, camera: &Camera) {
        let camera_pos = camera.get_position();
        self.fs_uniform.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, 1.0];
    }

    pub fn set_light_intensity(&mut self, light_intensity: f32) {
        self.fs_uniform.light_intensity = light_intensity;
    }
}

pub struct ObjectMaterial {
    texture: Arc<ImmutableImage<R8G8B8A8Unorm>>,
    sampler: Arc<Sampler>,
    ks: [f32; 3],
}

impl ObjectMaterial {
    pub fn new(mesh_renderer: &ObjectRenderer, texture: &RgbaImage, ks: [f32; 3]) -> Result<Self> {
        let (texture, texture_init) = ImmutableImage::from_iter(
            // FIXIT: read the picture vertically flipped otherwise the uv coordinates are incorrect
            texture.pixels().map(|p| p.0),
            Dimensions::Dim2d {
                width: texture.width(),
                height: texture.height(),
            },
            MipmapsCount::One,
            R8G8B8A8Unorm,
            mesh_renderer.get_queue(),
        )
        .chain_err(|| "fail to create texture for the texture")?;
        texture_init
            .then_signal_fence_and_flush()
            .chain_err(|| "fail to signal the fence and flush when initializing the texture image")?
            .wait(None)
            .chain_err(|| "fail to wait for the texture image being initialized")?;
        Ok(Self {
            texture,
            sampler: Sampler::simple_repeat_linear(mesh_renderer.get_device()),
            ks,
        })
    }
}

impl Material for ObjectMaterial {
    type Uniforms = ObjectUniforms;
    type Shaders = PhongShaders;

    fn create_uniforms(&self, device: Arc<Device>, queue: Arc<Queue>) -> Result<Self::Uniforms> {
        let vs_uniform_buffer = DeviceLocalBuffer::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            vec![queue.family()],
        )
        .chain_err(|| {
            "fail to create device local buffer to store the vertex shader uniform of the object \
            material"
        })?;
        let fs_uniform_buffer = DeviceLocalBuffer::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            vec![queue.family()],
        )
        .chain_err(|| {
            "fail to create device local buffer to store the fragment shader uniform of the object \
            material"
        })?;
        Ok(ObjectUniforms {
            vs_uniform: Default::default(),
            fs_uniform: FSUniform {
                kd: Default::default(),
                ks: [self.ks[0], self.ks[1], self.ks[2], 0.0],
                light_pos: Default::default(),
                camera_pos: Default::default(),
                light_intensity: Default::default(),
            },

            vs_uniform_buffer,
            fs_uniform_buffer,
            texture: self.texture.clone(),
            sampler: self.sampler.clone(),
        })
    }
}

pub type ObjectRenderer = MeshRenderer<ObjectVertex, ObjectMaterial>;

pub struct Object<S> {
    pub material: Arc<ObjectMaterial>,
    pub mesh: Mesh<ObjectVertex, ObjectMaterial, S>,
}

impl<S> Object<S> {
    pub fn new(
        mesh_renderer: Arc<ObjectRenderer>,
        position: &Vec<[f32; 3]>,
        texture_coord: &Vec<[f32; 2]>,
        normal: &Vec<[f32; 3]>,
        group: &Group,
        material: Arc<ObjectMaterial>,
    ) -> Result<Self> {
        let mut vertex_data: Vec<ObjectVertex> = vec![];
        // (position, texture_coord, normal)
        let mut vertex_data_to_index: HashMap<
            (
                [OrderedFloat<f32>; 3],
                [OrderedFloat<f32>; 3],
                [OrderedFloat<f32>; 2],
            ),
            usize,
        > = Default::default();
        let mut indices: Vec<u16> = vec![];
        for polygon in group.polys.iter() {
            for IndexTuple(position_index, texture_index, normal_index) in polygon.0.iter() {
                let normal_index =
                    normal_index.ok_or::<Error>("object without normals not supported".into())?;
                let texture_index =
                    texture_index.ok_or::<Error>("object without textures not supported".into())?;
                let position = position
                    .get(*position_index)
                    .ok_or::<Error>("fail to find position with given index".into())?;
                let normal = normal
                    .get(normal_index)
                    .ok_or::<Error>("fail to find normal with given index".into())?;
                let texture = texture_coord
                    .get(texture_index)
                    .ok_or::<Error>("fail to find texture coord with given index".into())?
                    .clone();
                // unwrap should be safe here
                let key = (
                    position
                        .iter()
                        .map(|v| OrderedFloat(*v))
                        .collect::<Vec<_>>()
                        .as_slice()
                        .try_into()
                        .unwrap(),
                    normal
                        .iter()
                        .map(|v| OrderedFloat(*v))
                        .collect::<Vec<_>>()
                        .as_slice()
                        .try_into()
                        .unwrap(),
                    texture
                        .iter()
                        .map(|v| OrderedFloat(*v))
                        .collect::<Vec<_>>()
                        .as_slice()
                        .try_into()
                        .unwrap(),
                );
                let i = match vertex_data_to_index.get(&key) {
                    Some(i) => *i,
                    None => {
                        let i = vertex_data.len();
                        vertex_data.push(ObjectVertex {
                            in_position: [position[0], position[1], position[2], 1.0],
                            in_normal: [normal[0], normal[1], normal[2], 0.0],
                            in_texture_coord: texture,
                        });
                        vertex_data_to_index.insert(key, i);
                        i
                    }
                };
                indices.push(i as u16);
            }
        }
        let mesh_data =
            MeshData::create(vertex_data, indices).chain_err(|| "fail to load vertex data")?;
        let mesh = mesh_renderer
            .create_mesh(mesh_data, material.as_ref())
            .chain_err(|| "fail to create mesh")?;
        Ok(Self { material, mesh })
    }
}
