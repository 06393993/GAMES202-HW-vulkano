// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

mod mesh_renderer;

use std::{path::PathBuf, sync::Arc};

use euclid::{Point3D, Transform3D};
use image::{io::Reader as ImageReader, RgbaImage};
use obj::{Obj, ObjData};
use vulkano::{
    command_buffer::{
        pool::standard::StandardCommandPoolBuilder, AutoCommandBufferBuilder, SubpassContents,
    },
    device::{Device, Queue},
    format::{ClearValue, D16Unorm, Format},
    framebuffer::{Framebuffer, RenderPassAbstract, Subpass},
    image::{attachment::AttachmentImage, traits::ImageViewAccess},
};

use super::{
    light::{PointLight, PointLightRenderer},
    material::{Material, UniformsT},
    object::{Object, ObjectMaterial, ObjectRenderer},
    Camera, TriangleSpace, WorldSpace,
};
use crate::errors::*;
pub use mesh_renderer::{Mesh, MeshData, MeshT, Renderer as MeshRenderer, SimpleVertex};

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

const LIGHT_INTENSITY: f32 = 1.0;

pub struct State {
    pub color: [f32; 3],
    pub camera: Camera,
    pub point_light_transform: Transform3D<f32, TriangleSpace, WorldSpace>,
    pub model_transform: Transform3D<f32, TriangleSpace, WorldSpace>,
}

pub struct Renderer {
    point_light: PointLight<TriangleSpace>,
    object_renderer: ObjectRenderer,
    objects: Vec<Object<TriangleSpace>>,
    depth_buffer: Arc<AttachmentImage<D16Unorm>>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
}

impl Renderer {
    pub fn init(
        device: Arc<Device>,
        queue: Arc<Queue>,
        format: Format,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let depth_format = Format::D16Unorm;
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        load: DontCare,
                        store: Store,
                        format: format,
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: Store,
                        format: depth_format,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            )
            .chain_err(|| "fail to create render pass when initializing renderer")?,
        );
        let subpass = Subpass::from(render_pass.clone(), 0)
            .expect("fail to retrieve the first subpass from the renderpass");
        let point_light_renderer = Arc::new(
            PointLightRenderer::init(
                device.clone(),
                queue.clone(),
                subpass.clone(),
                width,
                height,
            )
            .chain_err(|| "fail to create point light renderer")?,
        );
        let point_light = PointLight::new(
            point_light_renderer.clone(),
            LIGHT_INTENSITY,
            [1.0, 0.0, 0.0],
        )
        .chain_err(|| "fail to create point light")?;
        let object_renderer = ObjectRenderer::init(
            device.clone(),
            queue.clone(),
            subpass.clone(),
            width,
            height,
        )
        .chain_err(|| "fail to create object renderer")?;
        let depth_buffer = AttachmentImage::new(device.clone(), [width, height], D16Unorm)
            .chain_err(|| "fail to create the image for the depth attachment")?;
        Ok(Self {
            point_light,
            object_renderer,
            objects: vec![],
            depth_buffer,
            render_pass,
        })
    }

    pub fn load_model_and_texture(&mut self, model_and_texture: ModelAndTexture) -> Result<()> {
        let position = &model_and_texture.obj.position;
        let normal = &model_and_texture.obj.normal;
        let texture_coord = model_and_texture
            .obj
            .texture
            .iter()
            .map(|[u, v]| [*u, 1.0 - *v])
            .collect();
        let material = Arc::new(
            ObjectMaterial::with_texture(
                &self.object_renderer,
                model_and_texture.texture.as_ref(),
                // TODO: read ks from the obj file
                [0.0, 0.0, 0.0],
            )
            .chain_err(|| "fail to create the object material")?,
        );
        for object in model_and_texture.obj.objects.iter() {
            for group in object.groups.iter() {
                self.objects.push(
                    Object::with_texture(
                        self.object_renderer.clone(),
                        position,
                        &texture_coord,
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

    pub fn draw_commands(
        &self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<StandardCommandPoolBuilder>,
        image: Arc<impl ImageViewAccess + Send + Sync + 'static>,
        state: &State,
    ) -> Result<()> {
        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(image.clone())
                .chain_err(|| "fail to add the color attachment to the framebuffer")?
                .add(self.depth_buffer.clone())
                .chain_err(|| "fail to add the depth attachment to the framebuffer")?
                .build()
                .chain_err(|| "fail to create the framebuffer to draw on")?,
        );
        self.point_light
            .mesh
            .prepare_draw_commands(cmd_buf_builder, &state.point_light_transform, &state.camera)
            .chain_err(|| "fail to issue commands to prepare drawing for the point light mesh")?;
        for object in self.objects.iter() {
            {
                let mut uniforms = object.get_uniforms_lock();
                uniforms.set_light_pos(
                    state
                        .point_light_transform
                        .transform_point3d(Point3D::origin())
                        .ok_or::<Error>("invalid point light model transform".into())?,
                );
                uniforms.set_camera_pos(&state.camera);
                uniforms.set_light_intensity(LIGHT_INTENSITY);
            }
            object
                .mesh
                .prepare_draw_commands(cmd_buf_builder, &state.model_transform, &state.camera)
                .chain_err(|| "fail to issue commands to prepare drawing for the object mesh")?;
        }
        cmd_buf_builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::Inline,
                vec![ClearValue::None, ClearValue::Depth(1.0)],
            )
            .chain_err(|| "fail to add the begin renderpass command to the command builder")?;
        self.point_light
            .mesh
            .draw_commands(cmd_buf_builder)
            .chain_err(|| "fail to issue draw commands for the point light mesh")?;
        for object in self.objects.iter() {
            object
                .mesh
                .draw_commands(cmd_buf_builder)
                .chain_err(|| "fail to issue draw commands for the object mesh")?;
        }
        cmd_buf_builder
            .end_render_pass()
            .chain_err(|| "fail to add the end renderpass command to the command builder")?;
        Ok(())
    }
}
