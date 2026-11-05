use glam::Vec3;

/// Free-fly camera. Speed is a fixed value in world units per second, set with the
/// scroll wheel (see `move_speed` in main.rs).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_deg: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self { pos: Vec3::new(0.0, 0.0, -2.5), yaw: 0.0, pitch: 0.0, fov_deg: 60.0 }
    }

    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let fwd = Vec3::new(sy * cp, sp, cy * cp);
        let right = Vec3::Y.cross(fwd).normalize();
        let up = fwd.cross(right);
        (fwd, right, up)
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx;
        self.pitch = (self.pitch - dy).clamp(-1.55, 1.55);
    }
}
