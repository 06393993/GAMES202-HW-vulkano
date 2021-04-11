// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::{collections::HashMap, hash::Hash, marker::PhantomData, sync::Arc};

use euclid::{Point3D, Transform3D};
use image::RgbaImage;
use obj::{Group, IndexTuple};
use ordered_float::OrderedFloat;
use vulkano::{
    command_buffer::{pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder},
    device::{Device, Queue},
    format::R8G8B8A8Unorm,
    framebuffer::{RenderPassAbstract, Subpass},
    image::{immutable::ImmutableImage, Dimensions, MipmapsCount},
    pipeline::vertex::Vertex,
    sampler::Sampler,
    sync::GpuFuture,
};

use super::{
    light::PointLight,
    material::{Material, SetCamera},
    renderer::{Mesh, MeshData, MeshRenderer, MeshT},
    shaders::{
        phong::no_texture::{
            FsUniform as NoTexturePhongFsUniform, Shaders as NoTexturePhongShaders,
        },
        phong::with_texture::{FsUniform as TexturePhongFsUniform, Shaders as TexturePhongShaders},
        ShadersT, Texture, UniformsT,
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

pub struct TextureObjectMaterial {
    texture: Texture,
    ks: [f32; 3],
    kd: [f32; 3],
}

impl TextureObjectMaterial {
    pub fn new(renderer: &ObjectRenderer, texture: &RgbaImage, ks: [f32; 3]) -> Result<Self> {
        let mesh_renderer = &renderer.with_texture_renderer;
        let (image, image_init) = ImmutableImage::from_iter(
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
            texture: Texture {
                image,
                sampler: Sampler::simple_repeat_linear(mesh_renderer.get_device()),
            },
            kd: Default::default(),
            ks,
        })
    }
}

impl Material for TextureObjectMaterial {
    type Shaders = TexturePhongShaders;

    fn create_uniforms(
        &self,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<<TexturePhongShaders as ShadersT>::Uniforms> {
        <TexturePhongShaders as ShadersT>::Uniforms::new(
            device,
            queue,
            Default::default(),
            TexturePhongFsUniform {
                kd: [self.kd[0], self.kd[1], self.kd[2], 0.0],
                ks: [self.ks[0], self.ks[1], self.ks[2], 0.0],
                light_pos: Default::default(),
                camera_pos: Default::default(),
                light_intensity: Default::default(),
            },
            self.texture.clone(),
        )
    }
}

pub struct NoTextureObjectMaterial {
    ks: [f32; 3],
    kd: [f32; 3],
}

impl NoTextureObjectMaterial {
    pub fn new(kd: [f32; 3], ks: [f32; 3]) -> Result<Self> {
        Ok(Self { kd, ks })
    }
}

impl Material for NoTextureObjectMaterial {
    type Shaders = NoTexturePhongShaders;

    fn create_uniforms(
        &self,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<<NoTexturePhongShaders as ShadersT>::Uniforms> {
        <NoTexturePhongShaders as ShadersT>::Uniforms::new(
            device,
            queue,
            Default::default(),
            NoTexturePhongFsUniform {
                kd: [self.kd[0], self.kd[1], self.kd[2], 0.0],
                ks: [self.ks[0], self.ks[1], self.ks[2], 0.0],
                light_pos: Default::default(),
                camera_pos: Default::default(),
                light_intensity: Default::default(),
            },
        )
    }
}

#[derive(Clone)]
pub struct ObjectRenderer {
    with_texture_renderer: Arc<MeshRenderer<ObjectWithTextureVertex, TextureObjectMaterial>>,
    no_texture_renderer: Arc<MeshRenderer<ObjectWithNoTextureVertex, NoTextureObjectMaterial>>,
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
            MeshRenderer::init(device, queue, subpass, width, height)
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
    positions: &'a [[f32; 3]],
    textures: Option<&'a [[f32; 2]]>,
    normals: &'a [[f32; 3]],
) -> impl 'a + Fn(&'a IndexTuple) -> Result<(&'a [f32; 3], Option<&'a [f32; 2]>, Option<&'a [f32; 3]>)>
{
    move |IndexTuple(position_index, texture_index, normal_index)| {
        Ok((
            positions
                .get(*position_index)
                .ok_or_else(|| -> Error { "fail to find position with given index".into() })?,
            if let (Some(i), Some(textures)) = (texture_index, textures) {
                Some(textures.get(*i).ok_or_else(|| -> Error {
                    "fail to find texture coord with given index".into()
                })?)
            } else {
                None
            },
            normal_index
                .map(|i| {
                    normals
                        .get(i)
                        .ok_or_else(|| -> Error { "fail to find normal with given index".into() })
                })
                .transpose()?,
        ))
    }
}

pub trait ObjectUniforms: UniformsT + SetCamera {
    fn set_light_pos(&mut self, _light_pos: &Point3D<f32, WorldSpace>);
    fn set_camera_pos(&mut self, _camera: &Camera);
    fn set_light_intensity(&mut self, _light_intensity: f32);
}

impl ObjectUniforms for <NoTexturePhongShaders as ShadersT>::Uniforms {
    fn set_light_pos(&mut self, light_pos: &Point3D<f32, WorldSpace>) {
        self.fs_uniform.light_pos = [light_pos.x, light_pos.y, light_pos.z, 1.0];
    }

    fn set_camera_pos(&mut self, camera: &Camera) {
        let camera_pos = camera.get_position();
        self.fs_uniform.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, 1.0];
    }

    fn set_light_intensity(&mut self, light_intensity: f32) {
        self.fs_uniform.light_intensity = light_intensity;
    }
}

impl ObjectUniforms for <TexturePhongShaders as ShadersT>::Uniforms {
    fn set_light_pos(&mut self, light_pos: &Point3D<f32, WorldSpace>) {
        self.fs_uniform.light_pos = [light_pos.x, light_pos.y, light_pos.z, 1.0];
    }

    fn set_camera_pos(&mut self, camera: &Camera) {
        let camera_pos = camera.get_position();
        self.fs_uniform.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, 1.0];
    }

    fn set_light_intensity(&mut self, light_intensity: f32) {
        self.fs_uniform.light_intensity = light_intensity;
    }
}

struct VertexAttributes<'a> {
    position: &'a [[f32; 3]],
    texture_coord: Option<&'a [[f32; 2]]>,
    normal: &'a [[f32; 3]],
}

pub struct ObjectImpl<V: Vertex, M: Material, S> {
    mesh: Mesh<V, M, S>,
    uniforms: <<M as Material>::Shaders as ShadersT>::Uniforms,
}

type TextureObject<S> = ObjectImpl<ObjectWithTextureVertex, TextureObjectMaterial, S>;
type NoTextureObject<S> = ObjectImpl<ObjectWithNoTextureVertex, NoTextureObjectMaterial, S>;

impl<V: Vertex, M: Material, S> ObjectImpl<V, M, S>
where
    <<M as Material>::Shaders as ShadersT>::Uniforms: ObjectUniforms,
{
    fn new<K>(
        mesh_renderer: Arc<MeshRenderer<V, M>>,
        vertex_attributes: VertexAttributes<'_>,
        group: &Group,
        material: Arc<M>,
        vertex_to_struct: impl Fn(
            Result<(&[f32; 3], Option<&[f32; 2]>, Option<&[f32; 3]>)>,
        ) -> Result<V>,
        vertex_to_key: impl Fn(&V) -> K,
    ) -> Result<Self>
    where
        V: Vertex,
        K: Hash + Eq,
        M: Material + 'static,
        <<M as Material>::Shaders as ShadersT>::Uniforms: ObjectUniforms + SetCamera,
    {
        let VertexAttributes {
            position,
            texture_coord,
            normal,
        } = vertex_attributes;
        let vertex_data = group
            .polys
            .iter()
            .flat_map(|poly| poly.0.iter())
            .map(create_index_to_vertex_map(position, texture_coord, normal))
            .map(vertex_to_struct);
        let (vertex_data, indices) =
            vertex_attributes_to_indexed_vertex_attributes(vertex_data, vertex_to_key)
                .chain_err(|| "fail to generte indexed vertex attributes from vertex attributes")?;
        let mesh_data =
            MeshData::create(vertex_data, indices).chain_err(|| "fail to load vertex data")?;
        let (mesh, uniforms) = mesh_renderer
            .create_mesh(mesh_data, material.as_ref())
            .chain_err(|| "fail to create mesh")?;
        Ok(Self { mesh, uniforms })
    }
}

pub enum Object<S> {
    WithTexture(TextureObject<S>),
    NoTexture(NoTextureObject<S>),
}

impl<S> Object<S> {
    pub fn without_texture(
        renderer: ObjectRenderer,
        position: &[[f32; 3]],
        normal: &[[f32; 3]],
        group: &Group,
        material: Arc<NoTextureObjectMaterial>,
    ) -> Result<Self> {
        NoTextureObject::new(
            renderer.no_texture_renderer,
            VertexAttributes {
                position,
                texture_coord: None,
                normal,
            },
            group,
            material,
            |v| {
                let (position, _, normal) = v?;
                let normal = normal
                    .ok_or_else(|| -> Error { "object without normals not supported".into() })?;
                Ok(ObjectWithNoTextureVertex {
                    in_position: [position[0], position[1], position[2], 1.0],
                    in_normal: [normal[0], normal[1], normal[2], 0.0],
                })
            },
            |v| {
                (
                    Convert::<[f32; 4], _>::to(&v.in_position),
                    Convert::<[f32; 4], _>::to(&v.in_normal),
                )
            },
        )
        .chain_err(|| "fail to create an object without textures")
        .map(Self::NoTexture)
    }

    pub fn with_texture(
        renderer: ObjectRenderer,
        position: &[[f32; 3]],
        texture_coord: &[[f32; 2]],
        normal: &[[f32; 3]],
        group: &Group,
        material: Arc<TextureObjectMaterial>,
    ) -> Result<Self> {
        TextureObject::new(
            renderer.with_texture_renderer,
            VertexAttributes {
                position,
                texture_coord: Some(texture_coord),
                normal,
            },
            group,
            material,
            |v| {
                let (position, texture, normal) = v?;
                let normal = normal
                    .ok_or_else(|| -> Error { "object without normals not supported".into() })?;
                let texture = texture
                    .ok_or_else(|| -> Error { "object without textures not supported".into() })?;
                Ok(ObjectWithTextureVertex {
                    in_position: [position[0], position[1], position[2], 1.0],
                    in_normal: [normal[0], normal[1], normal[2], 0.0],
                    in_texture_coord: *texture,
                })
            },
            |v| {
                (
                    Convert::<[f32; 4], _>::to(&v.in_position),
                    Convert::<[f32; 4], _>::to(&v.in_normal),
                    Convert::<[f32; 2], _>::to(&v.in_texture_coord),
                )
            },
        )
        .chain_err(|| "fail to create an object with textures")
        .map(Self::WithTexture)
    }

    pub fn prepare_draw_commands<T>(
        &mut self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
        model_transform: &Transform3D<f32, S, WorldSpace>,
        camera: &Camera,
        light: &PointLight<T>,
    ) -> Result<()> {
        let uniforms: &mut dyn ObjectUniforms = match self {
            Self::WithTexture(ref mut obj) => &mut obj.uniforms,
            Self::NoTexture(ref mut obj) => &mut obj.uniforms,
        };
        uniforms.set_light_pos(
            &light
                .get_position()
                .chain_err(|| "fail to get light position")?,
        );
        uniforms.set_camera_pos(camera);
        uniforms.set_light_intensity(light.get_intensity());
        uniforms.set_model_matrix(model_transform.to_array());
        uniforms.set_view_proj_matrix_from_camera(camera);
        uniforms.update_buffers(cmd_buf_builder).chain_err(|| {
            "fail to add the update buffer for uniforms command to the command builder"
        })?;
        Ok(())
    }

    pub fn draw_commands(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
    ) -> Result<()> {
        let mesh: &dyn MeshT<S> = match self {
            Self::WithTexture(ref obj) => &obj.mesh,
            Self::NoTexture(ref obj) => &obj.mesh,
        };
        mesh.draw_commands(cmd_buf_builder)
    }
}
