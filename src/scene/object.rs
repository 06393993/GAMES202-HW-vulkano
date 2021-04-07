// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::{
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    sync::{Arc, Mutex, MutexGuard},
};

use euclid::Point3D;
use image::RgbaImage;
use obj::{Group, IndexTuple};
use ordered_float::OrderedFloat;
use vulkano::{
    buffer::{device_local::DeviceLocalBuffer, BufferUsage},
    command_buffer::{pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder},
    descriptor::{
        descriptor_set::{DescriptorSet, PersistentDescriptorSet},
        pipeline_layout::PipelineLayoutAbstract,
    },
    device::{Device, Queue},
    format::R8G8B8A8Unorm,
    framebuffer::{RenderPassAbstract, Subpass},
    image::{immutable::ImmutableImage, Dimensions, MipmapsCount},
    sampler::Sampler,
    sync::GpuFuture,
};

use super::{
    material::{Material, UniformsT},
    renderer::{MeshData, MeshRenderer, MeshT},
    shaders::{
        phong::NoTextureShaders as NoTexturePhongShaders,
        phong::TextureShaders as TexturePhongShaders, ShadersT,
    },
    Camera, WorldSpace,
};
use crate::errors::*;

#[derive(Default, Copy, Clone)]
pub struct ObjectWithTextureVertex {
    in_position: [f32; 4],
    in_normal: [f32; 4],
    in_texture_coord: [f32; 2],
}

vulkano::impl_vertex!(
    ObjectWithTextureVertex,
    in_position,
    in_normal,
    in_texture_coord
);

#[derive(Default, Copy, Clone)]
pub struct ObjectWithNoTextureVertex {
    in_position: [f32; 4],
    in_normal: [f32; 4],
}

vulkano::impl_vertex!(ObjectWithNoTextureVertex, in_position, in_normal);

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

#[derive(Clone)]
struct Texture {
    image: Arc<ImmutableImage<R8G8B8A8Unorm>>,
    sampler: Arc<Sampler>,
}

pub struct ObjectUniforms {
    vs_uniform: VSUniform,
    fs_uniform: FSUniform,

    vs_uniform_buffer: Arc<DeviceLocalBuffer<VSUniform>>,
    fs_uniform_buffer: Arc<DeviceLocalBuffer<FSUniform>>,
    texture: Option<Texture>,
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

    fn update_buffers(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
    ) -> Result<()> {
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
        let descriptor_set_builder = PersistentDescriptorSet::start(layout.clone())
                .add_buffer(self.vs_uniform_buffer.clone())
                .chain_err(|| "fail to add the vertex shader uniform buffer to the descriptor set for the object uniforms, binding = 0")?
                .add_buffer(self.fs_uniform_buffer.clone())
                .chain_err(|| "fail to add the fragment shader uniform buffer to the descriptor set for the object uniforms, binding = 1")?;
        let descriptor_set: Arc<dyn DescriptorSet + Send + Sync + 'static> = match self.texture {
            Some(ref texture) => Arc::new(descriptor_set_builder
                .add_sampled_image(texture.image.clone(), texture.sampler.clone())
                .chain_err(|| "fail to add the image with the sampler to the descriptor set for the object uniforms, binding = 2")?
                .build()
                .chain_err(|| "fail to create the descriptor set for the object uniforms")?
            ),
            None => Arc::new(
                descriptor_set_builder
                .build()
                .chain_err(|| "fail to create the descriptor set for the object uniforms")?
            ),
        };
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

pub struct ObjectMaterial<S> {
    texture: Option<Texture>,
    ks: [f32; 3],
    phantom: PhantomData<S>,
}

impl ObjectMaterial<TexturePhongShaders> {
    pub fn with_texture(
        renderer: &ObjectRenderer,
        texture: &RgbaImage,
        ks: [f32; 3],
    ) -> Result<Self> {
        let mesh_renderer = &renderer.with_texture_renderer;
        let (image, image_init) = ImmutableImage::from_iter(
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
        image_init
            .then_signal_fence_and_flush()
            .chain_err(|| "fail to signal the fence and flush when initializing the texture image")?
            .wait(None)
            .chain_err(|| "fail to wait for the texture image being initialized")?;
        Ok(Self {
            texture: Some(Texture {
                image,
                sampler: Sampler::simple_repeat_linear(mesh_renderer.get_device()),
            }),
            ks,
            phantom: PhantomData,
        })
    }
}

impl ObjectMaterial<NoTexturePhongShaders> {
    pub fn without_texture(ks: [f32; 3]) -> Result<Self> {
        Ok(Self {
            texture: None,
            ks,
            phantom: PhantomData,
        })
    }
}

impl<T: ShadersT> Material for ObjectMaterial<T> {
    type Uniforms = ObjectUniforms;
    type Shaders = T;

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
        })
    }
}

#[derive(Clone)]
pub struct ObjectRenderer {
    with_texture_renderer:
        Arc<MeshRenderer<ObjectWithTextureVertex, ObjectMaterial<TexturePhongShaders>>>,
    no_texture_renderer:
        Arc<MeshRenderer<ObjectWithNoTextureVertex, ObjectMaterial<NoTexturePhongShaders>>>,
}

impl ObjectRenderer {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass<impl RenderPassAbstract + Send + Sync + Clone + 'static>,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let with_texture_renderer = Arc::new(
            MeshRenderer::init(
                device.clone(),
                queue.clone(),
                subpass.clone(),
                width,
                height,
            )
            .chain_err(|| "fail to initialize renderer for object with textures")?,
        );
        let no_texture_renderer = Arc::new(
            MeshRenderer::init(
                device.clone(),
                queue.clone(),
                subpass.clone(),
                width,
                height,
            )
            .chain_err(|| "fail to initialize renderer for object without textures")?,
        );
        Ok(Self {
            with_texture_renderer,
            no_texture_renderer,
        })
    }
}

fn vertex_attributes_to_indexed_vertex_attributes<V, F, K>(
    vertices: impl Iterator<Item = Result<V>>,
    to_key: F,
) -> Result<(Vec<V>, Vec<u16>)>
where
    F: Fn(&V) -> K,
    K: Eq + Hash,
{
    let mut vertex2index: HashMap<K, u16> = Default::default();
    let mut res_indices = vec![];
    let mut res_vertices = vec![];
    for v in vertices {
        let v = v?;
        let key = to_key(&v);
        let i = match vertex2index.get(&key) {
            Some(i) => *i,
            None => {
                let i = res_vertices.len() as u16;
                res_vertices.push(v);
                vertex2index.insert(key, i);
                i
            }
        };
        res_indices.push(i as u16);
    }
    Ok((res_vertices, res_indices))
}

struct Convert<F, T>(PhantomData<(F, T)>);

impl Convert<[f32; 2], [OrderedFloat<f32>; 2]> {
    fn to(from: &[f32; 2]) -> [OrderedFloat<f32>; 2] {
        [OrderedFloat(from[0]), OrderedFloat(from[1])]
    }
}

impl Convert<[f32; 4], [OrderedFloat<f32>; 4]> {
    fn to(from: &[f32; 4]) -> [OrderedFloat<f32>; 4] {
        [
            OrderedFloat(from[0]),
            OrderedFloat(from[1]),
            OrderedFloat(from[2]),
            OrderedFloat(from[3]),
        ]
    }
}

fn create_index_to_vertex_map<'a>(
    positions: &'a Vec<[f32; 3]>,
    textures: Option<&'a Vec<[f32; 2]>>,
    normals: &'a Vec<[f32; 3]>,
) -> impl 'a + Fn(&'a IndexTuple) -> Result<(&'a [f32; 3], Option<&'a [f32; 2]>, Option<&'a [f32; 3]>)>
{
    move |IndexTuple(position_index, texture_index, normal_index)| {
        Ok((
            positions
                .get(*position_index)
                .ok_or::<Error>("fail to find position with given index".into())?,
            if let (Some(i), Some(textures)) = (texture_index, textures) {
                Some(
                    textures
                        .get(*i)
                        .ok_or::<Error>("fail to find texture coord with given index".into())?,
                )
            } else {
                None
            },
            normal_index
                .map(|i| {
                    normals
                        .get(i)
                        .ok_or::<Error>("fail to find normal with given index".into())
                })
                .map_or(Ok(None), |v| v.map(Some))?,
        ))
    }
}

pub struct Object<S> {
    pub mesh: Box<dyn MeshT<S>>,
    uniforms: Arc<Mutex<ObjectUniforms>>,
}

impl<S: 'static> Object<S> {
    pub fn with_texture(
        renderer: ObjectRenderer,
        position: &Vec<[f32; 3]>,
        texture_coord: &Vec<[f32; 2]>,
        normal: &Vec<[f32; 3]>,
        group: &Group,
        material: Arc<ObjectMaterial<TexturePhongShaders>>,
    ) -> Result<Self> {
        let mesh_renderer = renderer.with_texture_renderer;
        let vertex_data = group
            .polys
            .iter()
            .flat_map(|poly| poly.0.iter())
            .map(create_index_to_vertex_map(
                position,
                Some(texture_coord),
                normal,
            ))
            .map(|v| {
                let (position, texture, normal) = v?;
                let normal =
                    normal.ok_or::<Error>("object without normals not supported".into())?;
                let texture =
                    texture.ok_or::<Error>("object without textures not supported".into())?;
                Ok(ObjectWithTextureVertex {
                    in_position: [position[0], position[1], position[2], 1.0],
                    in_normal: [normal[0], normal[1], normal[2], 0.0],
                    in_texture_coord: texture.clone(),
                })
            });
        let (vertex_data, indices) =
            vertex_attributes_to_indexed_vertex_attributes(vertex_data, |v| {
                (
                    Convert::<[f32; 4], _>::to(&v.in_position),
                    Convert::<[f32; 4], _>::to(&v.in_normal),
                    Convert::<[f32; 2], _>::to(&v.in_texture_coord),
                )
            })
            .chain_err(|| "fail to generte indexed vertex attributes from vertex attributes")?;
        let mesh_data =
            MeshData::create(vertex_data, indices).chain_err(|| "fail to load vertex data")?;
        let mesh = mesh_renderer
            .create_mesh(mesh_data, material.as_ref())
            .chain_err(|| "fail to create mesh")?;
        let uniforms = mesh.uniforms();
        Ok(Self {
            mesh: Box::new(mesh),
            uniforms,
        })
    }

    pub fn without_texture(
        renderer: ObjectRenderer,
        position: &Vec<[f32; 3]>,
        normal: &Vec<[f32; 3]>,
        group: &Group,
        material: Arc<ObjectMaterial<NoTexturePhongShaders>>,
    ) -> Result<Self> {
        let mesh_renderer = renderer.no_texture_renderer;
        let vertex_data = group
            .polys
            .iter()
            .flat_map(|poly| poly.0.iter())
            .map(create_index_to_vertex_map(position, None, normal))
            .map(|v| {
                let (position, _, normal) = v?;
                let normal =
                    normal.ok_or::<Error>("object without normals not supported".into())?;
                Ok(ObjectWithNoTextureVertex {
                    in_position: [position[0], position[1], position[2], 1.0],
                    in_normal: [normal[0], normal[1], normal[2], 0.0],
                })
            });
        let (vertex_data, indices) =
            vertex_attributes_to_indexed_vertex_attributes(vertex_data, |v| {
                (
                    Convert::<[f32; 4], _>::to(&v.in_position),
                    Convert::<[f32; 4], _>::to(&v.in_normal),
                )
            })
            .chain_err(|| "fail to generte indexed vertex attributes from vertex attributes")?;
        let mesh_data =
            MeshData::create(vertex_data, indices).chain_err(|| "fail to load vertex data")?;
        let mesh = mesh_renderer
            .create_mesh(mesh_data, material.as_ref())
            .chain_err(|| "fail to create mesh")?;
        let uniforms = mesh.uniforms();
        Ok(Self {
            mesh: Box::new(mesh),
            uniforms,
        })
    }

    pub fn get_uniforms_lock(&self) -> MutexGuard<ObjectUniforms> {
        self.uniforms
            .lock()
            .expect("fail to grab the lock for uniforms")
    }
}
