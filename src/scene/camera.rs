use std::time::Duration;

use euclid::{approxeq::ApproxEq, point3, vec3, Angle, Point2D, Point3D, Transform3D, Vector3D};

use super::{NDCSpace, ViewSpace, WorldSpace};
use crate::errors::*;

#[derive(Clone)]
pub struct Camera {
    projection_transform: Transform3D<f32, ViewSpace, NDCSpace>,
    position: Point3D<f32, WorldSpace>,
    // a normalized vector from the camera position to the look at target
    direction: Vector3D<f32, WorldSpace>,
    up: Vector3D<f32, WorldSpace>,
}

impl Camera {
    pub fn new(
        fov: Angle<f32>,
        aspect_ratio: f32,
        near: f32,
        far: f32,
        position: &Point3D<f32, WorldSpace>,
        look_at: &Point3D<f32, WorldSpace>,
        up: &Vector3D<f32, WorldSpace>,
    ) -> Result<Self> {
        if fov.radians < Angle::approx_epsilon()
            || fov.radians > Angle::<f32>::pi().radians - Angle::<f32>::approx_epsilon()
        {
            return Err(
                format!("fov = {}, is not within the range of 0 and pi", fov.radians).into(),
            );
        }
        if far <= near {
            return Err(format!(
                "far should be greater than near, far = {}, near = {}",
                far, near
            )
            .into());
        }
        if near < f32::approx_epsilon() {
            return Err(format!("near should be greater than zero, near = {}", near).into());
        }
        if aspect_ratio < f32::approx_epsilon() {
            return Err(format!(
                "aspect ratio should be greater than zero, aspect ratio = {}",
                aspect_ratio
            )
            .into());
        }
        let direction = (*look_at - *position).normalize();
        let up = *up;

        if position.approx_eq(look_at) {
            return Err(format!(
                "camera look at target shouldn't be too close to the camera, \
                look at = {:?}, camera position = {:?}",
                look_at, position
            )
            .into());
        }
        if up.approx_eq(&Vector3D::zero()) {
            return Err("up shouldn't be zero".into());
        }
        if up.angle_to(direction).approx_eq(&Angle::zero()) {
            return Err(format!(
                "camera direction and up vector shouldn't be colinear, \
                up = {:?}, position = {:?}, look at = {:?}",
                up, position, look_at
            )
            .into());
        }

        let t = near * (fov / 2.0).radians.tan();
        let b = -t;
        let r = t * aspect_ratio;
        let l = -r;

        let projection_transform = Transform3D::from_arrays([
            [2.0 * near / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, -2.0 * near / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, -far / (far - near), -1.0],
            [0.0, 0.0, -far * near / (far - near), 0.0],
        ]);
        Ok(Camera {
            projection_transform,
            position: *position,
            direction,
            up,
        })
    }

    pub fn look_at(&mut self, target: &Point3D<f32, WorldSpace>) -> Result<()> {
        if target.approx_eq(&self.position) {
            return Err("the look at target is too close to the camera".into());
        }
        let direction = *target - self.position;
        if direction.angle_to(self.up).approx_eq(&Angle::zero()) {
            return Err(
                "the camera direction shouldn't be colinear to the up vector when setting look at \
                target"
                    .into(),
            );
        }
        self.direction = direction.normalize();
        Ok(())
    }

    pub fn set_position(&mut self, position: &Point3D<f32, WorldSpace>) {
        self.position = *position;
    }

    pub fn get_projection_transform(&self) -> Transform3D<f32, ViewSpace, NDCSpace> {
        self.projection_transform
    }

    pub fn get_view_transform(&self) -> Transform3D<f32, WorldSpace, ViewSpace> {
        // Schmidt orthogonalization
        let view_z = -self.direction;
        let view_y = (self.up - view_z * view_z.dot(self.up)).normalize();
        let view_x = view_y.cross(view_z).normalize();
        Transform3D::from_arrays([
            [view_x.x, view_y.x, view_z.x, 0.0],
            [view_x.y, view_y.y, view_z.y, 0.0],
            [view_x.z, view_y.z, view_z.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .pre_translate(-self.position.to_vector())
    }

    pub fn get_position(&self) -> Point3D<f32, WorldSpace> {
        self.position
    }

    pub fn get_direction(&self) -> Vector3D<f32, WorldSpace> {
        self.direction
    }

    pub fn get_aspect_ratio(&self) -> f32 {
        let proj = self.get_projection_transform();
        -proj.m22 / proj.m11
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    Forward,
    Backward,
}

pub trait CameraControl {
    fn get_camera_mut(&mut self) -> Result<&mut Camera>;
    // unit per second
    fn get_speed(&self) -> f32;

    fn move_camera(&mut self, direction: Direction, time_elapsed: Duration) -> Result<()> {
        let speed = self.get_speed();
        let camera = self
            .get_camera_mut()
            .chain_err(|| "fail to retrieve the camera")?;
        let pos = camera.get_position();
        let sign = match direction {
            Direction::Backward | Direction::Up | Direction::Right => 1.0,
            Direction::Forward | Direction::Down | Direction::Left => -1.0,
        };
        let view_transform_inverse = camera.get_view_transform().inverse().unwrap();
        let direction = match direction {
            Direction::Backward | Direction::Forward => vec3(0.0, 0.0, 1.0),
            Direction::Left | Direction::Right => vec3(1.0, 0.0, 0.0),
            Direction::Up | Direction::Down => vec3(0.0, 1.0, 0.0),
        };
        let direction = view_transform_inverse.transform_vector3d(direction) * sign;
        let dist = speed * time_elapsed.as_secs_f32();
        camera.set_position(&(pos + direction * dist));
        Ok(())
    }

    // the parameter is the target normalized camera direction projected on the x-y plane of the
    // view space
    fn rotate_camera_to(&mut self, projected_target: Point2D<f32, ViewSpace>) -> Result<()> {
        let projected_target = projected_target.to_vector();
        if projected_target.length() > 1.0 {
            return Err("the projected target is outside the unit circle".into());
        }
        let z = -(1.0 - projected_target.length()).sqrt();
        assert!(!z.is_nan());
        let target: Point3D<_, ViewSpace> = point3(projected_target.x, projected_target.y, z);
        let camera = self
            .get_camera_mut()
            .chain_err(|| "fail to retrieve camera")?;
        let target = camera
            .get_view_transform()
            .inverse()
            .expect("the inverse of the view transform should always exist")
            .transform_point3d(target)
            .expect("the inverse of the view transform should always make sense");
        camera
            .look_at(&target)
            .chain_err(|| format!("fail to set the camera look at target to {:?}", target))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_transform() {
        let near = 1.0;
        let far = 5.0;
        let camera = Camera::new(
            Angle::pi() / 3.0,
            2.0,
            near,
            far,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        let projection_transform = camera.get_projection_transform();
        assert!(projection_transform
            .transform_point3d_homogeneous(point3(0.0, 0.0, -near))
            .to_point3d()
            .unwrap()
            .approx_eq(&point3(0.0, 0.0, 0.0)));
        assert!(projection_transform
            .transform_point3d_homogeneous(point3(0.0, 0.0, -far))
            .to_point3d()
            .unwrap()
            .approx_eq(&point3(0.0, 0.0, 1.0)));
        assert!(projection_transform.approx_eq(&Transform3D::from_arrays([
            [3.0_f32.sqrt() / 2.0, 0.0, 0.0, 0.0],
            [0.0, -3.0_f32.sqrt(), 0.0, 0.0],
            [0.0, 0.0, -5.0 / 4.0, -1.0],
            [0.0, 0.0, -5.0 / 4.0, 0.0],
        ])));
    }

    #[test]
    fn test_fov_should_be_in_range_of_0_and_pi() {
        assert!(Camera::new(
            Angle::pi(),
            2.0,
            1.0,
            5.0,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());
        assert!(Camera::new(
            Angle::zero(),
            2.0,
            1.0,
            5.0,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());
    }

    #[test]
    fn test_far_should_be_greater_than_near() {
        assert!(Camera::new(
            Angle::pi() / 3.0,
            2.0,
            1.0,
            0.5,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());
    }

    #[test]
    fn test_near_should_be_greater_than_zero() {
        assert!(Camera::new(
            Angle::pi() / 3.0,
            2.0,
            0.0,
            0.5,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());

        assert!(Camera::new(
            Angle::pi() / 3.0,
            2.0,
            -1.0,
            0.5,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());
    }

    #[test]
    fn test_aspect_ratio_should_be_greater_than_zero() {
        assert!(Camera::new(
            Angle::pi() / 3.0,
            -2.0,
            1.0,
            5.0,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());
        assert!(Camera::new(
            Angle::pi() / 3.0,
            0.0,
            1.0,
            5.0,
            &point3(1.0, 0.0, 1.0),
            &Point3D::origin(),
            &vec3(0.0, 1.0, 0.0)
        )
        .is_err());
    }

    #[test]
    fn test_get_direction_and_position() {
        let position = point3(1.0, 0.0, 1.0);
        let direction = vec3(1.0, -1.0, 2.0);
        let camera = Camera::new(
            Angle::pi() / 3.0,
            2.0,
            1.0,
            5.0,
            &position,
            &(position + direction),
            &vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        let normalized_direction = camera.get_direction();

        assert!(normalized_direction
            .angle_to(direction)
            .approx_eq(&Angle::zero()));
        assert!(normalized_direction.length().approx_eq(&1.0));

        assert!(camera.get_position().approx_eq(&position));
    }

    #[test]
    fn test_get_view_transform() {
        let position = point3(1.0, 0.0, 1.0);
        let direction = vec3(1.0, -1.0, 2.0);
        let up = vec3(0.0, 1.0, 0.0);
        let camera = Camera::new(
            Angle::pi() / 3.0,
            2.0,
            1.0,
            5.0,
            &position,
            &(position + direction),
            &vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        let view_transform = camera.get_view_transform();
        let view_x = view_transform
            .inverse()
            .unwrap()
            .transform_vector3d(vec3(1.0, 0.0, 0.0));
        let view_y = view_transform
            .inverse()
            .unwrap()
            .transform_vector3d(vec3(0.0, 1.0, 0.0));
        let view_z = view_transform
            .inverse()
            .unwrap()
            .transform_vector3d(vec3(0.0, 0.0, 1.0));
        assert!(
            view_x.dot(direction).approx_eq(&0.0),
            "the x axis in view space should be perpendicular to the direction vector, dot product = {}", view_x.dot(direction)
        );
        assert!(
            view_x.dot(up).approx_eq(&0.0),
            "the x axis in view space should be perpendicular to the up vector"
        );
        assert!(
            view_x.length().approx_eq(&1.0),
            "the x axis in view space should share the same scale with the world space"
        );
        assert!(
            view_z.angle_to(direction).approx_eq(&Angle::pi()),
            "the z axis in view space should be inverse to the direction vector"
        );
        assert!(
            view_z.length().approx_eq(&1.0),
            "the z axis in view space should share the same scale with the world space"
        );
        assert!(
            view_y.dot(view_x).approx_eq(&0.0),
            "the y axis in view space should be perpendicular to both the x axis and the z axis"
        );
        assert!(
            view_y.dot(view_z).approx_eq(&0.0),
            "the y axis in view space should be perpendicular to both the x axis and the z axis"
        );
        assert!(
            view_y.length().approx_eq(&1.0),
            "the y axis in view space should share the same scale with the world space"
        );
        assert!(
            view_transform
                .inverse()
                .unwrap()
                .transform_point3d(Point3D::origin())
                .unwrap()
                .approx_eq(&position),
            "the origin point of the view space should be the camera position"
        );
    }

    #[test]
    fn test_direction_vector_and_up_vector_should_not_be_colinear() {
        let position = point3(1.0, 0.0, 1.0);
        let direction = vec3(1.0, -1.0, 2.0);
        let up = direction * 1.5;
        assert!(Camera::new(
            Angle::pi() / 3.0,
            2.0,
            1.0,
            5.0,
            &position,
            &(position + direction),
            &up,
        )
        .is_err());
    }

    #[test]
    fn test_position_and_look_at_target_should_not_be_too_close_at_initialization() {
        let position = point3(1.0, 0.0, 1.0);
        let up = vec3(0.0, 1.0, 0.0);
        assert!(Camera::new(Angle::pi() / 3.0, 2.0, 1.0, 5.0, &position, &position, &up,).is_err());
    }

    #[test]
    fn test_up_should_not_be_zero() {
        let position = point3(1.0, 0.0, 1.0);
        let direction = vec3(1.0, -1.0, 2.0);
        let up = Vector3D::zero();
        assert!(Camera::new(
            Angle::pi() / 3.0,
            2.0,
            1.0,
            5.0,
            &position,
            &(position + direction),
            &up,
        )
        .is_err());
    }

    #[test]
    fn test_position_and_look_at_target_should_not_be_too_close_when_setting_look_at_target() {
        let position = point3(1.0, 0.0, 1.0);
        let target = point3(2.0, -1.0, 3.0);
        let up = vec3(0.0, 1.0, 0.0);
        let mut camera =
            Camera::new(Angle::pi() / 3.0, 2.0, 1.0, 5.0, &position, &target, &up).unwrap();
        assert!(camera.look_at(&position).is_err());
    }

    #[test]
    fn test_view_transform_when_setting_position() {
        let fov = Angle::pi() / 3.0;
        let aspect_ratio = 2.0;
        let near = 1.0;
        let far = 5.0;
        let position = point3(1.0, 0.0, 1.0);
        let direction = vec3(1.0, -1.0, 2.0);
        let up = vec3(0.0, 1.0, 0.0);
        let mut camera1 = Camera::new(
            fov,
            aspect_ratio,
            near,
            far,
            &position,
            &(position + direction),
            &up,
        )
        .unwrap();

        let new_position = point3(2.0, 2.0, 4.0);
        camera1.set_position(&new_position);
        let camera2 = Camera::new(
            fov,
            aspect_ratio,
            near,
            far,
            &new_position,
            &(new_position + direction),
            &up,
        )
        .unwrap();
        assert!(camera1
            .get_view_transform()
            .approx_eq(&camera2.get_view_transform()));
    }

    #[test]
    fn test_view_transform_when_setting_look_at_target() {
        let fov = Angle::pi() / 3.0;
        let aspect_ratio = 2.0;
        let near = 1.0;
        let far = 5.0;
        let position = point3(1.0, 0.0, 1.0);
        let target = point3(2.0, -1.0, 3.0);
        let up = vec3(0.0, 1.0, 0.0);
        let mut camera1 =
            Camera::new(fov, aspect_ratio, near, far, &position, &target, &up).unwrap();

        let new_target = point3(3.0, 1.0, 6.0);
        camera1.look_at(&new_target);
        let camera2 =
            Camera::new(fov, aspect_ratio, near, far, &position, &new_target, &up).unwrap();
        assert!(camera1
            .get_view_transform()
            .approx_eq(&camera2.get_view_transform()));
    }

    #[test]
    fn test_up_should_not_be_colinear_with_direction_when_setting_look_at_target() {
        let fov = Angle::pi() / 3.0;
        let aspect_ratio = 2.0;
        let near = 1.0;
        let far = 5.0;
        let position = point3(1.0, 0.0, 1.0);
        let target = point3(2.0, -1.0, 3.0);
        let up = vec3(0.0, 1.0, 0.0);
        let mut camera =
            Camera::new(fov, aspect_ratio, near, far, &position, &target, &up).unwrap();
        assert!(camera.look_at(&(position + up)).is_err());
    }

    #[test]
    fn test_get_aspect_ratio() {
        let aspect_ratio = 2.5;
        let mut camera = Camera::new(
            Angle::pi() / 3.0,
            aspect_ratio,
            1.0,
            5.0,
            &point3(1.0, 0.0, 1.0),
            &point3(2.0, -1.0, 3.0),
            &vec3(0.0, 1.0, 0.0),
        )
        .unwrap();
        assert!(camera.get_aspect_ratio().approx_eq(&aspect_ratio));
    }
}
