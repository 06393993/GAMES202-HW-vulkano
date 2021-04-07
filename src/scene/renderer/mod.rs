// Copyright (c) 2021 06393993lky@gmail.com
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

mod mesh_renderer;

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use euclid::{Point3D, Transform3D};
use image::{io::Reader as ImageReader, RgbaImage};
use obj::{Obj, ObjData, ObjMaterial};
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
    textures: HashMap<String, Arc<RgbaImage>>,
}

impl ModelAndTexture {
    pub fn load(obj_path: &PathBuf) -> Result<Self> {
        let mut obj = Obj::load(obj_path.as_path()).chain_err(|| "fail to load obj file")?;
        obj.load_mtls()
            .chain_err(|| "fail to load associated mtl file")?;
        let mut textures: HashMap<_, _> = Default::default();
        for mtl in obj.data.material_libs.iter() {
            for material in mtl.materials.iter() {
                if let Some(ref name) = material.map_kd {
                    let texture_path = obj_path
                        .parent()
                        .expect("the path to obj file can't be root")
                        .join(&name);
                    let texture = ImageReader::open(texture_path.as_path())
                        .chain_err(|| {
                            format!("fail to open image file: {}", texture_path.display())
                        })?
                        .decode()
                        .chain_err(|| "fail to decode the image")?
                        .to_rgba8();
                    textures.insert(name.clone(), Arc::new(texture));
                }
            }
        }
        Ok(Self {
            obj: Arc::new(obj.data),
            textures,
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
        let mut name_to_texture_material: HashMap<_, _> = Default::default();
        let mut name_to_no_texture_material: HashMap<_, _> = Default::default();
        for mtl in model_and_texture.obj.material_libs.iter() {
            for material in mtl.materials.iter() {
                let name = &material.name;
                let ks = material.ks.unwrap_or([0.0, 0.0, 0.0]);
                if let Some(ref texture_name) = material.map_kd {
                    let texture = model_and_texture
                        .textures
                        .get(texture_name)
                        .ok_or::<Error>(
                            format!("fail to find map_kd with name {}", texture_name).into(),
                        )?
                        .clone();
                    if name_to_texture_material
                        .insert(
                            name,
                            Arc::new(
                                ObjectMaterial::with_texture(
                                    &self.object_renderer,
                                    texture.as_ref(),
                                    ks,
                                )
                                .chain_err(|| {
                                    format!("fail to create the object material {}", name)
                                })?,
                            ),
                        )
                        .is_some()
                    {
                        return Err(format!(
                            "materials with duplicate name {} not supproted",
                            name
                        )
                        .into());
                    };
                } else {
                    let kd = match material.kd {
                        Some(kd) => kd,
                        None => {
                            return Err(format!(
                                "the material {} with neither map_kd nor kd is not supported",
                                name
                            )
                            .into())
                        }
                    };
                    if name_to_no_texture_material
                        .insert(
                            name,
                            Arc::new(ObjectMaterial::without_texture(kd, ks).chain_err(|| {
                                format!("fail to create the object material {}", name)
                            })?),
                        )
                        .is_some()
                    {
                        return Err(format!(
                            "materials with duplicate name {} not supproted",
                            name
                        )
                        .into());
                    };
                }
            }
        }

        for object in model_and_texture.obj.objects.iter() {
            for group in object.groups.iter() {
                let material = match &group.material {
                    Some(ObjMaterial::Mtl(material)) => material,
                    Some(ObjMaterial::Ref(name)) => {
                        return Err(format!(
                            "object material {} in group {} not loaded",
                            name, group.name
                        )
                        .into())
                    }
                    None => {
                        return Err(format!(
                            "object group {} without material associated is not supported",
                            group.name
                        )
                        .into())
                    }
                };
                if material.map_kd.is_some() {
                    let material = name_to_texture_material
                        .get(&material.name)
                        .expect("all material should have been loaded");
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
                } else {
                    let material = name_to_no_texture_material
                        .get(&material.name)
                        .expect("all material should have been loaded");
                    self.objects.push(
                        Object::without_texture(
                            self.object_renderer.clone(),
                            position,
                            normal,
                            group,
                            material.clone(),
                        )
                        .chain_err(|| "fail to create object")?,
                    );
                }
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
