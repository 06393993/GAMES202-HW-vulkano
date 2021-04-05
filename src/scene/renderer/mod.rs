// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

mod mesh_renderer;

use std::{path::PathBuf, sync::Arc};

use euclid::Transform3D;
use image::{io::Reader as ImageReader, RgbaImage};
use obj::{Obj, ObjData};
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    device::{Device, Queue},
    format::Format,
    image::traits::ImageViewAccess,
};

use super::{
    light::{PointLight, PointLightRenderer},
    material::{Material, UniformsT},
    object::{Object, ObjectMaterial, ObjectRenderer},
    Camera, TriangleSpace, WorldSpace,
};
use crate::errors::*;
pub use mesh_renderer::{Mesh, MeshData, Renderer as MeshRenderer, SimpleVertex};

#[derive(Clone)]
pub struct ModelAndTexture {
    obj: Arc<ObjData>,
    texture: Arc<RgbaImage>,
}

impl ModelAndTexture {
    pub fn load(obj_path: &PathBuf, texture_path: &PathBuf) -> Result<Self> {
        let obj = Obj::load(obj_path.as_path()).chain_err(|| "fail to load obj file")?;
        let texture = ImageReader::open(texture_path.as_path())
            .chain_err(|| format!("fail to open image file: {}", texture_path.display()))?
            .decode()
            .chain_err(|| "fail to decode the image")?
            .to_rgba8();
        Ok(Self {
            obj: Arc::new(obj.data),
            texture: Arc::new(texture),
        })
    }
}

pub struct State {
    pub color: [f32; 3],
    pub camera: Camera,
    pub model_transform: Transform3D<f32, TriangleSpace, WorldSpace>,
}

pub struct Renderer {
    point_light_renderer: Arc<PointLightRenderer>,
    point_light: PointLight<TriangleSpace>,
    object_renderer: Arc<ObjectRenderer>,
    objects: Vec<Object<TriangleSpace>>,
}

impl Renderer {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let point_light_renderer = Arc::new(
            MeshRenderer::init(device.clone(), queue.clone(), format, width, height)
                .chain_err(|| "fail to create mesh renderer")?,
        );
        let point_light = PointLight::new(point_light_renderer.clone(), 1.0, [1.0, 0.0, 0.0])
            .chain_err(|| "fail to create point light")?;
        let object_renderer = Arc::new(
            ObjectRenderer::init(device, queue, format, width, height)
                .chain_err(|| "fail to create object renderer")?,
        );
        Ok(Self {
            point_light_renderer,
            point_light,
            object_renderer,
            objects: vec![],
        })
    }

    pub fn load_model_and_texture(&mut self, model_and_texture: ModelAndTexture) -> Result<()> {
        let position = &model_and_texture.obj.position;
        let normal = &model_and_texture.obj.normal;
        let texture_coord = &model_and_texture.obj.texture;
        let material = Arc::new(
            ObjectMaterial::new(
                self.object_renderer.as_ref(),
                model_and_texture.texture.as_ref(),
                // TODO: read ks from the obj file
                [0.0, 0.0, 0.0],
            )
            .chain_err(|| "fail to create the object material")?,
        );
        for object in model_and_texture.obj.objects.iter() {
            for group in object.groups.iter() {
                self.objects.push(
                    Object::new(
                        self.object_renderer.clone(),
                        position,
                        texture_coord,
                        normal,
                        group,
                        material.clone(),
                    )
                    .chain_err(|| "fail to create object")?,
                );
            }
        }
        Ok(())
    }

    pub fn draw_commands<P>(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<P>,
        image: Arc<impl ImageViewAccess + Send + Sync + 'static>,
        state: &State,
    ) -> Result<()> {
        let framebuffer = self
            .point_light_renderer
            .create_framebuffer(image)
            .chain_err(|| "fail to create framebuffer for light")?;
        self.point_light
            .mesh
            .draw_commands(
                cmd_buf_builder,
                framebuffer.clone(),
                &state.model_transform,
                &state.camera,
            )
            .chain_err(|| "fail to issue draw commands for the point light mesh")?;
        for object in self.objects.iter() {
            object
                .mesh
                .draw_commands(
                    cmd_buf_builder,
                    framebuffer.clone(),
                    &state.model_transform,
                    &state.camera,
                )
                .chain_err(|| "fail to issue draw commands for the object mesh")?;
        }
        Ok(())
    }
}
