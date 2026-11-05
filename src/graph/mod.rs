//! Node graphs.
//!
//! A scene has three graphs (`Graphs`), edited as separate modes of the node editor
//! like Blender's shader / world / compositor editors:
//! - **Material** (`GraphKind::Material`): the fractal itself, see below;
//! - **World** (`GraphKind::World`): sky, sun, ambient light and fog, ending in
//!   `World Output` (see `codegen::generate_world`);
//! - **Processing** (`GraphKind::Post`): image effects applied to the render, from
//!   `Render Layers` to `Image Output` (see `codegen::generate_post` and `post.rs`).
//!
//! The material canvas holds three graphs that are compiled separately (see `codegen`):
//! - the **domain graph**, ending in `Domain Output` (optional): transforms the point
//!   being rendered before iterating, e.g. `Repeat` tiles the fractal through space;
//! - the **iteration graph**, ending in `Iteration Output`: one step of the fractal
//!   loop, Mandelbulb3D-hybrid style. `Iteration Start` provides the incoming state
//!   (`z` plus the running derivative `dr`) and formula nodes transform it;
//! - the **material graph**, ending in `Material Output`: evaluated once per surface
//!   hit, from orbit traps and geometry, to produce the surface material (color,
//!   metallic, roughness, emission, subsurface).
//!
//! Sockets are typed (State, Float, Vector/Color). Every numeric input can be linked;
//! unlinked inputs use an inline value. Inline values live in a uniform array, so
//! editing them never recompiles; only structural edits (nodes, links, enum choices) do.

use crate::anim::{Interp, KEY_EPSILON};

pub mod codegen;
pub mod editor;
pub mod eval;
pub mod history;
pub mod looks;

pub type NodeId = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SocketType {
    /// Fractal iteration state (`z`, `dr`).
    State,
    Float,
    /// Vector or color (vec3).
    Vec3,
}

impl SocketType {
    /// Number of inline floats an unlinked input of this type stores.
    pub fn width(self) -> usize {
        match self {
            Self::State => 0,
            Self::Float => 1,
            Self::Vec3 => 3,
        }
    }

    /// Float and Vec3 convert implicitly (splat / average), like Blender.
    pub fn compatible(self, to: SocketType) -> bool {
        (self == Self::State) == (to == Self::State)
    }
}

/// The three graphs of a scene, each edited as its own mode of the node editor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum GraphKind {
    #[default]
    Material,
    World,
    Post,
}

impl GraphKind {
    pub const ALL: [GraphKind; 3] = [Self::Material, Self::World, Self::Post];

    pub fn label(self) -> &'static str {
        match self {
            Self::Material => "Material",
            Self::World => "World",
            Self::Post => "Processing",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Which compiled function a node may appear in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Context {
    Domain,
    Iteration,
    Color,
    /// The world graph: evaluated per view direction.
    World,
    /// The processing graph: evaluated per pixel of the rendered image.
    Post,
    Any,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Input,
    Output,
    Driver,
    Audio,
    Domain,
    Fractal,
    Fold,
    Transform,
    Combine,
    Sky,
    Adjust,
    Filter,
    Effect,
    Math,
    Utility,
    Texture,
    Color,
}

impl Category {
    pub const ALL: [Category; 17] = [
        Self::Input,
        Self::Output,
        Self::Driver,
        Self::Audio,
        Self::Domain,
        Self::Fractal,
        Self::Fold,
        Self::Transform,
        Self::Combine,
        Self::Sky,
        Self::Adjust,
        Self::Filter,
        Self::Effect,
        Self::Math,
        Self::Utility,
        Self::Texture,
        Self::Color,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Input => "Input",
            Self::Output => "Output",
            Self::Driver => "Driver",
            Self::Audio => "Audio",
            Self::Domain => "Domain",
            Self::Fractal => "Fractal",
            Self::Fold => "Fold",
            Self::Transform => "Transform",
            Self::Combine => "Combine",
            Self::Sky => "Sky",
            Self::Adjust => "Adjust",
            Self::Filter => "Filter",
            Self::Effect => "Effect",
            Self::Math => "Math",
            Self::Utility => "Utility",
            Self::Texture => "Texture",
            Self::Color => "Color",
        }
    }
}

pub struct InputDef {
    pub name: &'static str,
    pub ty: SocketType,
    pub default: [f32; 3],
    pub min: f32,
    pub max: f32,
    pub speed: f32,
    /// Non-empty: an enum setting (index into these labels). Not linkable, and
    /// baked into the generated code, so changing it recompiles.
    pub choices: &'static [&'static str],
    pub integer: bool,
    /// Vec3 shown as a color picker rather than three numbers.
    pub color: bool,
    /// Unlinked, this input reads the natural position of its graph (like a State input
    /// reads the incoming state) instead of an inline value: the domain point in the
    /// domain graph, `z` in the iteration graph, the surface hit in the material graph.
    pub implicit_position: bool,
    /// A setting read on the CPU (audio nodes): edited inline, never linked.
    pub knob: bool,
    /// RGB curves (master, R, G, B), edited with a curve widget: `CURVE_FLOATS` inline
    /// floats, never linked or keyframed.
    pub curve: bool,
}

/// Control points per curve of an RGB Curves input; endpoints sit at x = 0 and 1.
pub const CURVE_POINTS: usize = 5;
/// Curves per RGB Curves input: combined (C), R, G, B.
pub const CURVE_CHANNELS: usize = 4;
/// Inline floats of an RGB Curves input: (x, y) per point per channel.
pub const CURVE_FLOATS: usize = CURVE_POINTS * CURVE_CHANNELS * 2;

impl InputDef {
    pub fn linkable(&self) -> bool {
        if self.knob || self.curve {
            return false;
        }
        self.choices.is_empty()
    }

    /// Inline floats stored for this input (enum choices store their index).
    pub fn width(&self) -> usize {
        if self.implicit_position {
            0
        } else if self.curve {
            CURVE_FLOATS
        } else if self.choices.is_empty() {
            self.ty.width()
        } else {
            1
        }
    }

    /// Shown as a plain label rather than an editable value when unlinked.
    pub fn label_only(&self) -> bool {
        self.ty == SocketType::State || self.implicit_position
    }

    /// Height in node-editor rows (the curve widget is taller).
    pub fn rows(&self) -> f32 {
        if self.curve { 7.0 } else { 1.0 }
    }

    /// Inline values of a fresh node: the defaults, or identity curves.
    pub fn default_values(&self) -> Vec<f32> {
        if self.curve {
            let n = (CURVE_POINTS - 1) as f32;
            return (0..CURVE_CHANNELS * CURVE_POINTS).flat_map(|k| [(k % CURVE_POINTS) as f32 / n; 2]).collect();
        }
        self.default[..self.width()].to_vec()
    }

    /// Can be keyframed: inline numbers (not an enum, which would need a recompile per
    /// change, a curve, or a state / position input).
    pub fn keyable(&self) -> bool {
        self.choices.is_empty() && !self.curve && self.width() > 0
    }
}

const BASE: InputDef = InputDef {
    name: "",
    ty: SocketType::Float,
    default: [0.0; 3],
    min: f32::NEG_INFINITY,
    max: f32::INFINITY,
    speed: 0.005,
    choices: &[],
    integer: false,
    color: false,
    implicit_position: false,
    knob: false,
    curve: false,
};

const fn st(name: &'static str) -> InputDef {
    InputDef { name, ty: SocketType::State, ..BASE }
}
const fn fl(name: &'static str, default: f32, min: f32, max: f32, speed: f32) -> InputDef {
    InputDef { name, default: [default, 0.0, 0.0], min, max, speed, ..BASE }
}
/// Unbounded float.
const fn num(name: &'static str, default: f32) -> InputDef {
    fl(name, default, f32::NEG_INFINITY, f32::INFINITY, 0.01)
}
const fn int(name: &'static str, default: f32, min: f32, max: f32) -> InputDef {
    InputDef { name, default: [default, 0.0, 0.0], min, max, speed: 0.05, integer: true, ..BASE }
}
const fn ch(name: &'static str, default: f32, choices: &'static [&'static str]) -> InputDef {
    InputDef {
        name,
        default: [default, 0.0, 0.0],
        min: 0.0,
        max: (choices.len() - 1) as f32,
        choices,
        integer: true,
        ..BASE
    }
}
const fn vec(name: &'static str, x: f32, y: f32, z: f32) -> InputDef {
    InputDef { name, ty: SocketType::Vec3, default: [x, y, z], ..BASE }
}
/// Position input that defaults to the domain position when unlinked.
const fn pos(name: &'static str) -> InputDef {
    InputDef { name, ty: SocketType::Vec3, implicit_position: true, ..BASE }
}
/// CPU-side setting (see `InputDef::knob`).
const fn knob(name: &'static str, default: f32, min: f32, max: f32, speed: f32) -> InputDef {
    InputDef { name, default: [default, 0.0, 0.0], min, max, speed, knob: true, ..BASE }
}
const fn knob_int(name: &'static str, default: f32, min: f32, max: f32) -> InputDef {
    InputDef { integer: true, speed: 0.05, ..knob(name, default, min, max, 0.05) }
}
const fn col(name: &'static str, r: f32, g: f32, b: f32) -> InputDef {
    InputDef { name, ty: SocketType::Vec3, default: [r, g, b], min: 0.0, max: 1.0, color: true, ..BASE }
}
/// RGB curves (see `InputDef::curve`).
const fn curves(name: &'static str) -> InputDef {
    InputDef { name, curve: true, ..BASE }
}

pub struct OutputDef {
    pub name: &'static str,
    pub ty: SocketType,
}

const fn out(name: &'static str, ty: SocketType) -> OutputDef {
    OutputDef { name, ty }
}

const STATE_OUT: &[OutputDef] = &[out("State", SocketType::State)];
const FLOAT_OUT: &[OutputDef] = &[out("Value", SocketType::Float)];
const COLOR_OUT: &[OutputDef] = &[out("Color", SocketType::Vec3)];
const VECTOR_OUT: &[OutputDef] = &[out("Vector", SocketType::Vec3)];

pub const MATH_OPS: &[&str] = &[
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Power",
    "Minimum",
    "Maximum",
    "Modulo",
    "Sine",
    "Cosine",
    "Absolute",
    "Fraction",
    "Floor",
    "Square Root",
    "Logarithm",
    "Exponent",
    "Tangent",
    "Arctangent2",
    "Sign",
    "Ping-pong",
    "Snap",
    "Round",
];
pub const VECTOR_OPS: &[&str] = &["Add", "Subtract", "Multiply", "Scale", "Cross", "Normalize", "Length", "Dot"];
/// Audio Level bands; must match `audio::Band` order.
pub const AUDIO_BANDS: &[&str] = &["Full", "Bass", "Low mids", "Mids", "Highs", "Air"];
/// Beat triggers; must match `audio::Trigger` order.
pub const BEAT_TRIGGERS: &[&str] = &["Beat", "Bar", "Kick", "Snare", "Hi-hat"];
/// Floats at the top of the `gp` uniform reserved for live (CPU-computed) node
/// outputs, see `Graph::live_slots`.
pub const LIVE_SLOTS: usize = 32;
pub const WAVEFORMS: &[&str] = &["Sine", "Triangle", "Sawtooth", "Square"];
const PLAY_MODES: &[&str] = &["Once", "Loop", "Ping-pong"];
pub const BULB_VARIANTS: &[&str] = &["Standard", "Cosine (White)", "Burning Ship", "Mandelbar"];
pub const POLYHEDRA: &[&str] = &["Tetrahedral", "Octahedral", "Icosahedral"];
const EASINGS: &[&str] = &["Linear", "Smooth"];
const OFF_ON: &[&str] = &["off", "on"];
pub const AXES: &[&str] = &["X", "Y", "Z"];
pub const WAVE_TYPES: &[&str] = &["Bands X", "Bands Y", "Bands Z", "Rings", "Spiral"];
/// Distance estimators (Iteration Output). Kleinian and planar suit space-filling
/// fractals and make the scene unbounded.
pub const DE_MODES: &[&str] = &["log (bulbs)", "linear (folds)", "kleinian", "planar (Apollonian)"];
pub const DE_KLEINIAN: u32 = 2;
pub const DE_PLANAR: u32 = 3;
/// Image Output view transforms; must match `view_transform` in composite.wgsl.
pub const VIEW_TRANSFORMS: &[&str] = &["ACES filmic", "AgX", "Reinhard", "Standard (clip)"];
/// Blend node modes; must match `blend` in post.wgsl.
pub const BLEND_MODES: &[&str] = &[
    "Mix",
    "Add",
    "Subtract",
    "Multiply",
    "Screen",
    "Overlay",
    "Difference",
    "Lighten",
    "Darken",
    "Soft light",
    "Divide",
    "Color dodge",
];
/// Image Texture edge handling outside 0..1.
pub const EXTENSIONS: &[&str] = &["Repeat", "Extend", "Clip"];

/// World Output input indices.
pub const WORLD_COLOR: usize = 0;
pub const WORLD_STRENGTH: usize = 1;
pub const WORLD_AMBIENT: usize = 2;
pub const WORLD_SUN_AZIMUTH: usize = 3;
pub const WORLD_SUN_ELEVATION: usize = 4;
pub const WORLD_SUN_COLOR: usize = 5;
pub const WORLD_SUN_STRENGTH: usize = 6;
pub const WORLD_SUN_SIZE: usize = 7;
pub const WORLD_FOG: usize = 8;
/// Image Output input index of the view transform.
pub const POST_VIEW_TRANSFORM: usize = 1;

/// Material Output input index of Roughness.
pub const MAT_ROUGHNESS: usize = 2;
/// Must match the roughness cutoff for reflection rays in fractal.wgsl.
pub const REFLECTION_MAX_ROUGHNESS: f32 = 0.4;

/// Iteration Output input indices (input 0 is the State socket).
pub const OUT_BAILOUT: usize = 1;
pub const OUT_BOUND: usize = 2;
pub const OUT_DE_MODE: usize = 3;

macro_rules! node_kinds {
    ($($kind:ident),* $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
        pub enum NodeKind { $($kind),* }
        impl NodeKind {
            pub const ALL: &[NodeKind] = &[$(NodeKind::$kind),*];
        }
    };
}

node_kinds!(
    // Input / output
    IterStart, IterInfo, OrbitData, Geometry, Fresnel, Value, IterOutput, ColorOutput,
    // Drivers (values over time)
    Time, Oscillator, Animate, Wiggle,
    // Audio (values from the loaded song, computed on the CPU each frame)
    AudioLevel, Beat,
    // Domain (space before iterating)
    Position, Repeat, PolarRepeat, DomainOutput,
    // Fractal formulas (fold + scale + add c in one node)
    Mandelbulb, Mandelbox, AmazingSurf, Bristorbrot, QuickDudley, Makin3d, PineTree, Kaliset,
    Menger, Sierpinski, Kifs, PolyIfs, PseudoKleinian, Apollonian, Xenodreambuie,
    // Folds (distance-preserving: dr unchanged)
    BoxFold, SphereFold, AbsFold, OctaFold, TetraFold, PolyFold, PlaneFold, FractFold,
    PolarFold, SmoothAbsFold, CylinderFold,
    // Transforms
    Scale, IfsScale, Rotate, Translate, Twist, AddC, SphereInvert,
    // Combine
    Mix, Alternate, IterSwitch,
    // Math / utility
    Math, VectorMath, MapRange, Clamp, SeparateXyz, CombineXyz, RotateVector,
    // Textures (procedural, hash-based: identical on the CPU and GPU)
    NoiseTexture, VoronoiTexture, WaveTexture, WhiteNoise,
    // Color
    CosinePalette, ColorRamp, MixColor, Hsv, Blackbody, HueSaturation, Blend, RgbToBw,
    // World
    WorldOutput, SkyGradient, PhysicalSky, EnvTexture, StarField, Direction, SunInfo,
    // Processing: inputs / output
    RenderLayers, ImageTexture, ScreenCoords, PostOutput,
    // Processing: per-pixel adjustments
    Exposure, BrightContrast, Curves, ColorBalance, Invert, Gamma, Posterize, Vignette, FilmGrain,
    // Processing: effects that sample their input image around the pixel
    LensDistortion, Pixelate, Sharpen, Kaleidoscope,
    // Processing: multi-pass filters
    Blur, Bloom, MotionBlur,
);

/// How a processing node reads its first (Image) input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageInput {
    /// Per pixel, fused into the surrounding pass.
    Pixel,
    /// Sampled around the pixel: the input is rendered to a texture first.
    Sampled,
    /// Filtered in passes of its own (blur, bloom).
    Pass,
}

impl NodeKind {
    pub fn label(self) -> &'static str {
        use NodeKind::*;
        match self {
            IterStart => "Iteration Start",
            IterInfo => "Iteration Info",
            OrbitData => "Orbit Data",
            Geometry => "Geometry",
            Fresnel => "Fresnel",
            Value => "Value",
            IterOutput => "Iteration Output",
            ColorOutput => "Material Output",
            Time => "Time",
            Oscillator => "Oscillator",
            Animate => "Animate",
            Wiggle => "Wiggle",
            AudioLevel => "Audio Level",
            Beat => "Beat",
            Position => "Position",
            Repeat => "Repeat (modulo)",
            PolarRepeat => "Polar Repeat",
            DomainOutput => "Domain Output",
            Mandelbulb => "Mandelbulb",
            Mandelbox => "Mandelbox",
            AmazingSurf => "Amazing Surf",
            Bristorbrot => "Bristorbrot",
            QuickDudley => "Quick Dudley",
            Makin3d => "Makin 3D",
            PineTree => "Pine Tree (Benesi)",
            PolyIfs => "Polyhedral IFS",
            PseudoKleinian => "Pseudo-Kleinian",
            Apollonian => "Apollonian",
            Xenodreambuie => "Xenodreambuie",
            FractFold => "Fract Fold",
            PolyFold => "Polyhedral Fold",
            Kaliset => "Kaliset",
            Menger => "Menger Sponge",
            Sierpinski => "Sierpinski Tetrahedron",
            Kifs => "Kaleidoscopic IFS",
            BoxFold => "Box Fold",
            SphereFold => "Sphere Fold",
            AbsFold => "Abs Fold",
            OctaFold => "Octahedral Fold",
            TetraFold => "Tetrahedral Fold",
            PlaneFold => "Plane Fold",
            PolarFold => "Polar Fold",
            SmoothAbsFold => "Smooth Abs Fold",
            CylinderFold => "Cylinder Fold",
            Scale => "Scale",
            IfsScale => "IFS Scale",
            Rotate => "Rotate",
            Translate => "Translate",
            Twist => "Twist",
            AddC => "Add C",
            SphereInvert => "Sphere Inversion",
            Mix => "Mix",
            Alternate => "Alternate (hybrid)",
            IterSwitch => "Iteration Switch",
            Math => "Math",
            VectorMath => "Vector Math",
            MapRange => "Map Range",
            Clamp => "Clamp",
            SeparateXyz => "Separate XYZ",
            CombineXyz => "Combine XYZ",
            RotateVector => "Rotate Vector",
            CosinePalette => "Cosine Palette",
            ColorRamp => "Color Ramp",
            MixColor => "Mix Color",
            Hsv => "HSV to Color",
            Blackbody => "Blackbody",
            HueSaturation => "Hue / Saturation",
            NoiseTexture => "Noise Texture",
            VoronoiTexture => "Voronoi Texture",
            WaveTexture => "Wave Texture",
            WhiteNoise => "White Noise",
            Blend => "Blend",
            RgbToBw => "RGB to BW",
            WorldOutput => "World Output",
            SkyGradient => "Sky Gradient",
            PhysicalSky => "Physical Sky",
            EnvTexture => "Environment Texture",
            StarField => "Star Field",
            Direction => "Direction",
            SunInfo => "Sun Info",
            RenderLayers => "Render Layers",
            ImageTexture => "Image Texture",
            ScreenCoords => "Screen Coordinates",
            PostOutput => "Image Output",
            Exposure => "Exposure",
            BrightContrast => "Brightness / Contrast",
            Curves => "RGB Curves",
            ColorBalance => "Color Balance",
            Invert => "Invert",
            Gamma => "Gamma",
            Posterize => "Posterize",
            Vignette => "Vignette",
            FilmGrain => "Film Grain",
            LensDistortion => "Lens Distortion",
            Pixelate => "Pixelate",
            Sharpen => "Sharpen",
            Kaleidoscope => "Kaleidoscope",
            Blur => "Blur",
            Bloom => "Bloom",
            MotionBlur => "Motion Blur",
        }
    }

    pub fn category(self) -> Category {
        use NodeKind::*;
        match self {
            IterStart | IterInfo | OrbitData | Geometry | Fresnel | Value | Direction | SunInfo | RenderLayers
            | ImageTexture | ScreenCoords => Category::Input,
            IterOutput | ColorOutput | DomainOutput | WorldOutput | PostOutput => Category::Output,
            SkyGradient | PhysicalSky | EnvTexture | StarField => Category::Sky,
            Exposure | BrightContrast | Curves | ColorBalance | Invert | Gamma | Posterize => Category::Adjust,
            Blur | Bloom | Sharpen | MotionBlur => Category::Filter,
            Vignette | FilmGrain | LensDistortion | Pixelate | Kaleidoscope => Category::Effect,
            Blend | RgbToBw => Category::Color,
            Time | Oscillator | Animate | Wiggle => Category::Driver,
            AudioLevel | Beat => Category::Audio,
            Position | Repeat | PolarRepeat => Category::Domain,
            Mandelbulb | Mandelbox | AmazingSurf | Bristorbrot | QuickDudley | Makin3d | PineTree | Kaliset
            | Menger | Sierpinski | Kifs | PolyIfs | PseudoKleinian | Apollonian | Xenodreambuie => Category::Fractal,
            BoxFold | SphereFold | AbsFold | OctaFold | TetraFold | PolyFold | PlaneFold | FractFold | PolarFold
            | SmoothAbsFold | CylinderFold => Category::Fold,
            Scale | IfsScale | Rotate | Translate | Twist | AddC | SphereInvert => Category::Transform,
            Mix | Alternate | IterSwitch => Category::Combine,
            Math | VectorMath => Category::Math,
            MapRange | Clamp | SeparateXyz | CombineXyz | RotateVector => Category::Utility,
            NoiseTexture | VoronoiTexture | WaveTexture | WhiteNoise => Category::Texture,
            CosinePalette | ColorRamp | MixColor | Hsv | Blackbody | HueSaturation => Category::Color,
        }
    }

    pub fn context(self) -> Context {
        use NodeKind::*;
        match self {
            OrbitData | Geometry | Fresnel | ColorOutput => Context::Color,
            Position | Repeat | DomainOutput => Context::Domain,
            IterStart | IterInfo | IterOutput => Context::Iteration,
            WorldOutput | SkyGradient | PhysicalSky | EnvTexture | StarField | Direction | SunInfo => Context::World,
            RenderLayers | ImageTexture | ScreenCoords | PostOutput | Exposure | BrightContrast | Curves | ColorBalance
            | Invert | Gamma | Posterize | Vignette | FilmGrain | LensDistortion | Pixelate | Sharpen | Kaleidoscope
            | Blur | Bloom | MotionBlur => Context::Post,
            _ if self.outputs().iter().any(|o| o.ty == SocketType::State) => Context::Iteration,
            _ => Context::Any,
        }
    }

    /// Can be added to (and pasted into) a graph of this kind.
    pub fn allowed_in(self, g: GraphKind) -> bool {
        match self.context() {
            Context::Domain | Context::Iteration | Context::Color => g == GraphKind::Material,
            Context::World => g == GraphKind::World,
            Context::Post => g == GraphKind::Post,
            Context::Any => true,
        }
    }

    /// How a processing node reads its Image input (see `ImageInput`).
    pub fn image_input(self) -> ImageInput {
        use NodeKind::*;
        match self {
            LensDistortion | Pixelate | Sharpen | Kaleidoscope => ImageInput::Sampled,
            Blur | Bloom => ImageInput::Pass,
            _ => ImageInput::Pixel,
        }
    }

    /// Reads an image file (picked in the node; stored in `Node::file`).
    pub fn has_file(self) -> bool {
        matches!(self, NodeKind::EnvTexture | NodeKind::ImageTexture)
    }

    /// The output node every graph of its kind has; it can't be added or deleted.
    pub fn required(self) -> bool {
        matches!(self, NodeKind::IterOutput | NodeKind::WorldOutput | NodeKind::PostOutput)
    }

    /// Outputs computed on the CPU each frame and uploaded to live slots: the audio
    /// nodes, and Wiggle (its fBm would otherwise run in every iteration of every
    /// distance estimate, for a value that is the same for the whole frame).
    pub fn is_live(self) -> bool {
        matches!(self, NodeKind::AudioLevel | NodeKind::Beat | NodeKind::Wiggle)
    }

    /// Shown in the add menu (output nodes every graph has are always present).
    pub fn addable(self) -> bool {
        !self.required()
    }

    /// At most one per graph.
    pub fn unique(self) -> bool {
        matches!(
            self,
            NodeKind::IterOutput
                | NodeKind::ColorOutput
                | NodeKind::DomainOutput
                | NodeKind::WorldOutput
                | NodeKind::PostOutput
                | NodeKind::MotionBlur
        )
    }

    /// Can be muted: a formula node whose first input and output are State, so
    /// bypassing it just passes the state through; or a processing node that passes
    /// its Image straight through.
    pub fn mutable(self) -> bool {
        let first_in = self.inputs().first().filter(|i| i.linkable()).map(|i| i.ty);
        let first_out = self.outputs().first().map(|o| o.ty);
        let state = first_in == Some(SocketType::State) && first_out == Some(SocketType::State);
        let image = self.context() == Context::Post && first_in == Some(SocketType::Vec3) && first_out == Some(SocketType::Vec3);
        (state || image) && !self.required()
    }

    pub fn description(self) -> &'static str {
        use NodeKind::*;
        match self {
            IterStart => "State entering this iteration",
            IterInfo => "Iteration index, |z|, derivative, z itself and the point c (iteration graph only)",
            OrbitData => "Orbit traps collected while iterating (color graph only)",
            Geometry => "Surface position, normal, camera distance and facing (material graph only)",
            Fresnel => "Schlick Fresnel for an index of refraction: 0 facing the camera, 1 at grazing angles",
            Value => "A constant number",
            IterOutput => "State passed to the next iteration, plus loop settings",
            ColorOutput => "Surface material: base color, metallic, roughness, specular, emission, subsurface",
            DomainOutput => "Position fed to the fractal; connect Repeat here to tile space",
            Time => "Animation time in seconds (Animation controls in the Debug panel)",
            Oscillator => "Periodic wave over time: center + amplitude * wave",
            Animate => "Ramp from one value to another over a duration",
            Wiggle => "Smooth random motion over time: center + amplitude * noise (like an F-curve noise modifier). Computed once per frame",
            AudioLevel => "Loudness of a frequency band of the song, 0..1 (Audio panel loads the song)",
            Beat => "Fires on each beat / bar / drum hit of the song. Pulse: 1 then fades over decay.                      Count: hits so far. Phase: 0..1 to the next hit. Glide: count, eased over decay",
            Position => "The point being rendered (domain graph only)",
            Repeat => "Tiles space with modulo so the fractal repeats; limit 0 = infinite",
            PolarRepeat => "Radial tiling: folds the angle around an axis into one of n sectors. Sector = which one",
            RotateVector => "Rotate a vector by XYZ angles in degrees",
            Mandelbulb => "Spherical power formula, + c. Variants: Cosine, Burning Ship, Mandelbar",
            QuickDudley => "Quadratic triplex formula (Mandelbulb3D)",
            Makin3d => "Makin's quadratic 3D Mandelbrot (Mandelbulb3D)",
            PineTree => "Benesi's quadratic 'pine tree' formula",
            PolyIfs => "Knighty-style polyhedral kaleidoscopic IFS (tetra / octa / icosahedral)",
            PseudoKleinian => "Knighty's pseudo-Kleinian: box fold + clamped inversion. Use the 'kleinian' distance",
            Apollonian => "Apollonian gasket: periodic fold + sphere inversion. Use the 'planar' distance",
            Xenodreambuie => "Mandelbulb with its spherical angles offset by alpha / beta (Mandelbulb3D's Xenodreambuie)",
            PolarFold => "n-fold radial symmetry around an axis. Best in fold / IFS chains: before a + c formula it can stop points escaping (use Polar Repeat in the domain instead)",
            SmoothAbsFold => "Abs fold with a rounded crease (sqrt(z² + k²)): softer, organic folds",
            CylinderFold => "Sphere fold measured from an axis instead of a point: tubes and rings",
            FractFold => "Periodic fold: tiles space into cells of the given half-size",
            PolyFold => "Mirror into a tetra / octa / icosahedral fundamental domain",
            Mandelbox => "Box fold, sphere fold, scale, + c",
            AmazingSurf => "Mandelbox variant that only box-folds x and y",
            Bristorbrot => "Quadratic polynomial formula, + c",
            Kaliset => "abs(z) / |z|² inversion minus an offset",
            Menger => "Menger sponge IFS",
            Sierpinski => "Sierpinski tetrahedron IFS",
            Kifs => "Octahedral kaleidoscopic IFS with rotation",
            OctaFold => "Mirror into the octahedral fundamental domain",
            TetraFold => "Mirror into the tetrahedral fundamental domain",
            PlaneFold => "Mirror across a plane",
            IfsScale => "Scale about a center point (IFS stretch)",
            Twist => "Rotate around Y proportionally to height (approximate DE)",
            SphereInvert => "Inversion in a sphere",
            Mix => "Blend two branches (both are evaluated)",
            Alternate => "A for n iterations, then B for m, repeating",
            IterSwitch => "A before the switch iteration, B from it on",
            NoiseTexture => "Gradient (Perlin) noise with fBm detail. Unlinked vector = the graph's position",
            VoronoiTexture => "Cellular noise: distance to the nearest cell point, edge distance, per-cell color",
            WaveTexture => "Bands, rings or spirals, optionally distorted by noise",
            WhiteNoise => "Random value / color per input value: e.g. a different look per Repeat cell",
            Blackbody => "Color of a glowing body at a temperature in Kelvin (great for emission)",
            HueSaturation => "Shift hue, scale saturation and value of a color",
            Blend => "Combine two colors with a blend mode (Add, Multiply, Screen, Overlay, ...)",
            RgbToBw => "Luminance of a color (Rec. 709 weights)",
            WorldOutput => "Sky color seen in the background and lighting the fractal, plus the sun (direct light \
                            with shadows) and distance fog",
            SkyGradient => "Ground, horizon and zenith colors blended by view height",
            PhysicalSky => "Atmospheric scattering lit by the World Output's sun: blue sky, white haze, red sunsets",
            EnvTexture => "HDR panorama (equirectangular .hdr / .exr, or any image) around the scene",
            StarField => "Points of light scattered over the sky",
            Direction => "The view direction being shaded (world graph only)",
            SunInfo => "The World Output's sun: direction, height above the horizon (-1..1) and color",
            RenderLayers => "The rendered fractal: color, distance from the camera, and a mask of the fractal \
                             (1) against the sky (0)",
            ImageTexture => "A picture file, placed over the screen (unlinked vector = screen UV)",
            ScreenCoords => "Pixel position: UV 0..1, Centered (-aspect..aspect by -1..1, y up), distance from \
                             the center and angle",
            PostOutput => "The final image, and how it's mapped to the display (view transform)",
            Exposure => "Brightness in photographic stops: +1 doubles, -1 halves",
            BrightContrast => "Lift / lower brightness and stretch contrast around middle grey",
            Curves => "Tone curves on the combined image (C) and each channel; in display-like (gamma 2.2) space",
            ColorBalance => "Lift (shadows), gamma (midtones) and gain (highlights); grey is neutral",
            Invert => "Negative image",
            Gamma => "Power curve: above 1 darkens midtones, below 1 brightens them",
            Posterize => "Reduce to a number of levels per channel",
            Vignette => "Darken (or tint) toward the edges of the frame",
            FilmGrain => "Animated film-like noise",
            LensDistortion => "Barrel (+) / pincushion (-) distortion and chromatic aberration (dispersion)",
            Pixelate => "Blocky pixels of the given size",
            Sharpen => "Unsharp mask: boosts fine detail",
            Kaleidoscope => "Mirror the image into rotating wedges around the center",
            Blur => "Gaussian blur; radius in % of the image height",
            Bloom => "Glow around everything brighter than the threshold (emission, highlights, the sun)",
            MotionBlur => "Camera and animation motion blur for video exports: each frame accumulates its samples \
                           over the shutter (0..1 of a frame). Passes the image through in the viewport",
            _ => "",
        }
    }

    pub fn outputs(self) -> &'static [OutputDef] {
        use NodeKind::*;
        use SocketType::*;
        const IMAGE_OUT: &[OutputDef] = &[out("Image", Vec3)];
        match self {
            IterOutput | ColorOutput | DomainOutput | WorldOutput | PostOutput => &[],
            Blend | SkyGradient | PhysicalSky | EnvTexture | StarField => COLOR_OUT,
            RgbToBw => FLOAT_OUT,
            Direction => const { &[out("Direction", Vec3)] },
            SunInfo => const { &[out("Direction", Vec3), out("Height", Float), out("Color", Vec3)] },
            RenderLayers => const { &[out("Image", Vec3), out("Depth", Float), out("Mask", Float)] },
            ImageTexture => const { &[out("Color", Vec3), out("Alpha", Float)] },
            ScreenCoords => const {
                &[out("UV", Vec3), out("Centered", Vec3), out("Radius", Float), out("Angle", Float)]
            },
            Bloom => const { &[out("Image", Vec3), out("Glow", Vec3)] },
            Exposure | BrightContrast | Curves | ColorBalance | Invert | Gamma | Posterize | Vignette | FilmGrain
            | LensDistortion | Pixelate | Sharpen | Kaleidoscope | Blur | MotionBlur => IMAGE_OUT,
            Time | Oscillator | Animate | Wiggle => FLOAT_OUT,
            AudioLevel => const { &[out("Level", Float)] },
            Beat => const { &[out("Pulse", Float), out("Count", Float), out("Phase", Float), out("Glide", Float)] },
            Position => const { &[out("Position", Vec3)] },
            Repeat => const { &[out("Position", Vec3), out("Cell", Vec3)] },
            PolarRepeat => const { &[out("Position", Vec3), out("Sector", Float)] },
            NoiseTexture => const { &[out("Fac", Float), out("Color", Vec3)] },
            VoronoiTexture => const {
                &[out("Distance", Float), out("Edge", Float), out("Color", Vec3), out("Random", Float)]
            },
            WaveTexture => const { &[out("Fac", Float)] },
            WhiteNoise => const { &[out("Value", Float), out("Color", Vec3)] },
            RotateVector => VECTOR_OUT,
            IterInfo => const {
                &[out("Index", Float), out("Radius", Float), out("Derivative", Float), out("Z", Vec3), out("C", Vec3)]
            },
            OrbitData => const {
                &[
                    out("Trap radius", Float),
                    out("Trap X", Float),
                    out("Trap Y", Float),
                    out("Trap Z", Float),
                    out("Iterations", Float),
                ]
            },
            Geometry => const {
                &[out("Position", Vec3), out("Normal", Vec3), out("Distance", Float), out("Facing", Float)]
            },
            Fresnel => FLOAT_OUT,
            Value | Math | MapRange | Clamp => FLOAT_OUT,
            VectorMath => const { &[out("Vector", Vec3), out("Value", Float)] },
            SeparateXyz => const { &[out("X", Float), out("Y", Float), out("Z", Float)] },
            CombineXyz => VECTOR_OUT,
            CosinePalette | ColorRamp | MixColor | Hsv | Blackbody | HueSaturation => COLOR_OUT,
            _ => STATE_OUT,
        }
    }

    pub fn inputs(self) -> &'static [InputDef] {
        use NodeKind::*;
        const CW: InputDef = fl("c weight", 1.0, -2.0, 2.0, 0.005);
        const BOX: &[InputDef] = &[
            st("State"),
            fl("scale", 2.0, -4.0, 4.0, 0.005),
            fl("min radius", 0.5, 0.0, 2.0, 0.005),
            fl("fixed radius", 1.0, 0.0, 3.0, 0.005),
            fl("fold limit", 1.0, 0.0, 3.0, 0.005),
            CW,
        ];
        match self {
            IterStart | IterInfo | OrbitData | Geometry => &[],
            Value => const { &[num("value", 1.0)] },
            IterOutput => const {
                &[
                    st("State"),
                    fl("bailout", 4.0, 1.0, 4096.0, 0.5),
                    fl("bound radius", 1.5, 0.1, 64.0, 0.01),
                    ch("distance", 0.0, DE_MODES),
                ]
            },
            ColorOutput => const {
                &[
                    col("Base Color", 0.35, 0.45, 0.75),
                    fl("Metallic", 0.0, 0.0, 1.0, 0.005),
                    fl("Roughness", 0.4, 0.0, 1.0, 0.005),
                    fl("Specular", 0.5, 0.0, 1.0, 0.005),
                    col("Emission", 1.0, 0.5, 0.2),
                    fl("Emission Strength", 0.0, 0.0, 100.0, 0.05),
                    fl("Subsurface", 0.0, 0.0, 1.0, 0.005),
                    col("Subsurface Color", 0.9, 0.4, 0.3),
                    fl("Subsurface Radius", 0.5, 0.0, 1.0, 0.005),
                ]
            },
            Fresnel => const { &[fl("IOR", 1.45, 1.0, 5.0, 0.005)] },
            DomainOutput => const { &[pos("Position"), fl("view distance", 40.0, 1.0, 1000.0, 0.5)] },
            Time => const { &[num("speed", 1.0), num("offset", 0.0)] },
            Oscillator => const {
                &[
                    ch("waveform", 0.0, WAVEFORMS),
                    fl("frequency (Hz)", 0.2, 0.0, 50.0, 0.005),
                    num("amplitude", 1.0),
                    num("center", 0.0),
                    fl("phase", 0.0, 0.0, 1.0, 0.005),
                ]
            },
            Animate => const {
                &[
                    num("from", 0.0),
                    num("to", 1.0),
                    fl("duration (s)", 4.0, 0.01, 3600.0, 0.05),
                    fl("start (s)", 0.0, 0.0, 3600.0, 0.05),
                    ch("mode", 2.0, PLAY_MODES),
                    ch("easing", 1.0, EASINGS),
                ]
            },
            AudioLevel => const {
                &[
                    ch("band", 0.0, AUDIO_BANDS),
                    knob("attack (s)", 0.02, 0.0, 2.0, 0.002),
                    knob("release (s)", 0.25, 0.0, 4.0, 0.005),
                    knob("gain", 1.0, 0.0, 8.0, 0.01),
                ]
            },
            Beat => const {
                &[
                    ch("trigger", 0.0, BEAT_TRIGGERS),
                    knob_int("every", 1.0, 1.0, 64.0),
                    knob("decay (s)", 0.3, 0.0, 8.0, 0.005),
                    knob("nudge (s)", 0.0, -1.0, 1.0, 0.002),
                ]
            },
            Wiggle => const {
                &[
                    knob("frequency (Hz)", 0.5, 0.0, 50.0, 0.005),
                    knob("amplitude", 1.0, f32::NEG_INFINITY, f32::INFINITY, 0.01),
                    knob("center", 0.0, f32::NEG_INFINITY, f32::INFINITY, 0.01),
                    knob("detail", 2.0, 0.0, 8.0, 0.02),
                    knob_int("seed", 0.0, 0.0, 9999.0),
                ]
            },
            Position => &[],
            PolarRepeat => const {
                &[pos("Position"), fl("segments", 6.0, 1.0, 64.0, 0.02), ch("axis", 2.0, AXES), ch("mirror", 1.0, OFF_ON)]
            },
            Repeat => const {
                &[pos("Position"), vec("cell size", 3.0, 3.0, 3.0), vec("limit (0 = ∞)", 0.0, 0.0, 0.0), ch("mirror", 0.0, OFF_ON)]
            },
            RotateVector => const { &[vec("Vector", 0.0, 0.0, 0.0), vec("angles°", 0.0, 0.0, 0.0)] },
            Mandelbulb => const { &[st("State"), fl("power", 8.0, 1.0, 32.0, 0.01), CW, ch("variant", 0.0, BULB_VARIANTS)] },
            QuickDudley | Makin3d | PineTree => const { &[st("State"), CW] },
            PolyIfs => const {
                &[
                    st("State"),
                    ch("symmetry", 2.0, POLYHEDRA),
                    fl("scale", 2.0, 1.0, 4.0, 0.005),
                    vec("vertex / edge / face", 1.0, 0.0, 0.0),
                    fl("angle°", 0.0, -180.0, 180.0, 0.25),
                ]
            },
            PolyFold => const { &[st("State"), ch("symmetry", 2.0, POLYHEDRA)] },
            PseudoKleinian => const {
                &[st("State"), vec("box size", 0.92436, 0.90756, 0.92436), fl("inversion", 1.0, 0.1, 3.0, 0.002)]
            },
            Apollonian => const { &[st("State"), fl("inversion", 1.25, 0.5, 2.0, 0.002), vec("cell", 1.0, 1.0, 1.0)] },
            FractFold => const { &[st("State"), vec("half size", 1.0, 1.0, 1.0)] },
            Xenodreambuie => const {
                &[
                    st("State"),
                    fl("power", 8.0, 1.0, 32.0, 0.01),
                    fl("alpha°", 0.0, -180.0, 180.0, 0.25),
                    fl("beta°", 0.0, -180.0, 180.0, 0.25),
                    CW,
                ]
            },
            PolarFold => const {
                &[st("State"), fl("segments", 6.0, 1.0, 64.0, 0.02), ch("axis", 2.0, AXES), ch("mirror", 1.0, OFF_ON)]
            },
            SmoothAbsFold => const { &[st("State"), fl("smoothness", 0.1, 0.0, 2.0, 0.002)] },
            CylinderFold => const {
                &[
                    st("State"),
                    fl("min radius", 0.5, 0.0, 2.0, 0.005),
                    fl("fixed radius", 1.0, 0.0, 3.0, 0.005),
                    ch("axis", 2.0, AXES),
                ]
            },
            Mandelbox | AmazingSurf => BOX,
            Bristorbrot => const { &[st("State"), CW] },
            Kaliset => const {
                &[st("State"), fl("scale", 1.0, 0.1, 3.0, 0.005), vec("offset", 0.5, 0.5, 0.5), CW]
            },
            Menger => const { &[st("State"), fl("scale", 3.0, 1.0, 4.0, 0.005), vec("offset", 1.0, 1.0, 1.0)] },
            Sierpinski => const { &[st("State"), fl("scale", 2.0, 1.0, 4.0, 0.005), fl("offset", 1.0, 0.0, 2.0, 0.005)] },
            Kifs => const {
                &[st("State"), fl("scale", 2.0, 1.0, 4.0, 0.005), vec("offset", 1.0, 0.5, 0.25), fl("angle°", 15.0, -180.0, 180.0, 0.25)]
            },
            BoxFold => const { &[st("State"), fl("limit", 1.0, 0.0, 3.0, 0.005)] },
            SphereFold => const {
                &[st("State"), fl("min radius", 0.5, 0.0, 2.0, 0.005), fl("fixed radius", 1.0, 0.0, 3.0, 0.005)]
            },
            AbsFold => const { &[st("State"), ch("x", 1.0, OFF_ON), ch("y", 1.0, OFF_ON), ch("z", 1.0, OFF_ON)] },
            OctaFold | TetraFold => const { &[st("State")] },
            PlaneFold => const { &[st("State"), vec("normal", 1.0, 0.0, 0.0), fl("offset", 0.0, -4.0, 4.0, 0.005)] },
            Scale => const { &[st("State"), fl("scale", 2.0, -4.0, 4.0, 0.005)] },
            IfsScale => const { &[st("State"), fl("scale", 2.0, -4.0, 4.0, 0.005), vec("center", 1.0, 1.0, 1.0)] },
            Rotate => const { &[st("State"), vec("angles°", 0.0, 0.0, 0.0)] },
            Translate => const { &[st("State"), vec("offset", 0.0, 0.0, 0.0)] },
            Twist => const { &[st("State"), fl("amount", 0.5, -8.0, 8.0, 0.005)] },
            AddC => const { &[st("State"), fl("weight", 1.0, -2.0, 2.0, 0.005)] },
            SphereInvert => const { &[st("State"), fl("radius", 1.0, 0.01, 4.0, 0.005)] },
            Mix => const { &[st("A"), st("B"), fl("factor", 0.5, 0.0, 1.0, 0.005)] },
            Alternate => const {
                &[st("A"), st("B"), int("A iterations", 1.0, 0.0, 16.0), int("B iterations", 1.0, 0.0, 16.0)]
            },
            IterSwitch => const { &[st("A"), st("B"), int("switch at", 2.0, 0.0, 64.0)] },
            Math => const { &[ch("operation", 0.0, MATH_OPS), num("A", 0.0), num("B", 0.0)] },
            VectorMath => const {
                &[ch("operation", 0.0, VECTOR_OPS), vec("A", 0.0, 0.0, 0.0), vec("B", 0.0, 0.0, 0.0), num("scale", 1.0)]
            },
            MapRange => const {
                &[
                    num("value", 0.5),
                    num("from min", 0.0),
                    num("from max", 1.0),
                    num("to min", 0.0),
                    num("to max", 1.0),
                    ch("clamp", 1.0, OFF_ON),
                ]
            },
            Clamp => const { &[num("value", 0.5), num("min", 0.0), num("max", 1.0)] },
            SeparateXyz => const { &[vec("Vector", 0.0, 0.0, 0.0)] },
            CombineXyz => const { &[num("X", 0.0), num("Y", 0.0), num("Z", 0.0)] },
            CosinePalette => const {
                &[
                    num("factor", 0.5),
                    col("brightness", 0.5, 0.5, 0.5),
                    col("contrast", 0.5, 0.5, 0.5),
                    vec("frequency", 1.0, 1.0, 1.0),
                    vec("phase", 0.0, 0.1, 0.2),
                ]
            },
            ColorRamp => const {
                &[
                    num("factor", 0.5),
                    col("start", 0.05, 0.05, 0.15),
                    col("middle", 0.8, 0.35, 0.15),
                    col("end", 1.0, 0.95, 0.8),
                    fl("middle position", 0.5, 0.01, 0.99, 0.005),
                ]
            },
            MixColor => const { &[fl("factor", 0.5, 0.0, 1.0, 0.005), col("A", 0.2, 0.3, 0.6), col("B", 0.9, 0.6, 0.3)] },
            Hsv => const {
                &[num("hue", 0.0), fl("saturation", 0.7, 0.0, 1.0, 0.005), fl("value", 0.8, 0.0, 1.0, 0.005)]
            },
            Blackbody => const { &[fl("temperature (K)", 1800.0, 800.0, 40000.0, 10.0)] },
            HueSaturation => const {
                &[
                    col("Color", 0.8, 0.3, 0.2),
                    num("hue shift", 0.0),
                    fl("saturation", 1.0, 0.0, 4.0, 0.005),
                    fl("value", 1.0, 0.0, 8.0, 0.005),
                ]
            },
            NoiseTexture => const {
                &[
                    pos("Vector"),
                    fl("scale", 4.0, 0.0, 1000.0, 0.02),
                    fl("detail", 3.0, 0.0, 8.0, 0.02),
                    fl("roughness", 0.5, 0.0, 1.0, 0.005),
                    fl("distortion", 0.0, 0.0, 10.0, 0.005),
                    int("seed", 0.0, 0.0, 9999.0),
                ]
            },
            VoronoiTexture => const {
                &[
                    pos("Vector"),
                    fl("scale", 4.0, 0.0, 1000.0, 0.02),
                    fl("randomness", 1.0, 0.0, 1.0, 0.005),
                    int("seed", 0.0, 0.0, 9999.0),
                ]
            },
            WaveTexture => const {
                &[
                    pos("Vector"),
                    ch("type", 0.0, WAVE_TYPES),
                    ch("profile", 0.0, WAVEFORMS),
                    fl("scale", 4.0, 0.0, 1000.0, 0.02),
                    fl("distortion", 0.0, 0.0, 10.0, 0.005),
                    num("phase", 0.0),
                ]
            },
            WhiteNoise => const { &[pos("Vector"), int("seed", 0.0, 0.0, 9999.0)] },
            Blend => const {
                &[ch("mode", 0.0, BLEND_MODES), fl("factor", 1.0, 0.0, 1.0, 0.005), col("A", 0.2, 0.3, 0.6), col("B", 0.9, 0.6, 0.3)]
            },
            RgbToBw => const { &[col("Color", 0.8, 0.8, 0.8)] },

            WorldOutput => const {
                &[
                    col("Color", 0.10, 0.13, 0.20),
                    fl("Strength", 1.0, 0.0, 100.0, 0.005),
                    fl("Ambient", 2.0, 0.0, 16.0, 0.005),
                    fl("Sun azimuth°", 45.8, -360.0, 360.0, 0.25),
                    fl("Sun elevation°", 40.1, -90.0, 90.0, 0.25),
                    col("Sun color", 1.0, 0.92, 0.77),
                    fl("Sun strength", 1.3, 0.0, 100.0, 0.005),
                    fl("Sun size°", 0.53, 0.0, 20.0, 0.005),
                    fl("Fog density", 0.02, 0.0, 4.0, 0.001),
                ]
            },
            SkyGradient => const {
                &[
                    pos("Vector"),
                    col("Zenith", 0.10, 0.13, 0.20),
                    col("Horizon", 0.06, 0.075, 0.1175),
                    col("Ground", 0.02, 0.02, 0.035),
                    fl("sharpness", 1.0, 0.05, 16.0, 0.005),
                ]
            },
            PhysicalSky => const {
                &[
                    pos("Vector"),
                    fl("air", 1.0, 0.0, 10.0, 0.005),
                    fl("haze", 1.0, 0.0, 50.0, 0.01),
                    fl("altitude (km)", 0.5, 0.0, 60.0, 0.05),
                    col("Ground", 0.25, 0.22, 0.2),
                    fl("strength", 1.0, 0.0, 100.0, 0.005),
                ]
            },
            EnvTexture => const {
                &[pos("Vector"), fl("rotation°", 0.0, -360.0, 360.0, 0.5), fl("strength", 1.0, 0.0, 100.0, 0.005)]
            },
            StarField => const {
                &[
                    pos("Vector"),
                    fl("density", 0.3, 0.0, 1.0, 0.002),
                    fl("scale", 60.0, 1.0, 1000.0, 0.5),
                    fl("brightness", 4.0, 0.0, 1000.0, 0.02),
                    fl("twinkle", 0.0, 0.0, 1.0, 0.005),
                    int("seed", 0.0, 0.0, 9999.0),
                ]
            },
            Direction | SunInfo | RenderLayers | ScreenCoords => &[],
            ImageTexture => const {
                &[pos("Vector"), fl("scale", 1.0, 0.001, 1000.0, 0.005), ch("extension", 1.0, EXTENSIONS)]
            },

            PostOutput => const { &[col("Image", 0.0, 0.0, 0.0), ch("view transform", 0.0, VIEW_TRANSFORMS)] },
            Exposure => const { &[col("Image", 0.8, 0.8, 0.8), fl("stops", 0.0, -16.0, 16.0, 0.01)] },
            BrightContrast => const {
                &[col("Image", 0.8, 0.8, 0.8), fl("brightness", 0.0, -1.0, 1.0, 0.002), fl("contrast", 0.0, -1.0, 4.0, 0.002)]
            },
            Curves => const { &[col("Image", 0.8, 0.8, 0.8), fl("factor", 1.0, 0.0, 1.0, 0.005), curves("curves")] },
            ColorBalance => const {
                &[
                    col("Image", 0.8, 0.8, 0.8),
                    fl("factor", 1.0, 0.0, 1.0, 0.005),
                    col("Lift", 0.5, 0.5, 0.5),
                    col("Gamma", 0.5, 0.5, 0.5),
                    col("Gain", 0.5, 0.5, 0.5),
                ]
            },
            Invert => const { &[col("Image", 0.8, 0.8, 0.8), fl("factor", 1.0, 0.0, 1.0, 0.005)] },
            Gamma => const { &[col("Image", 0.8, 0.8, 0.8), fl("gamma", 1.0, 0.01, 10.0, 0.005)] },
            Posterize => const { &[col("Image", 0.8, 0.8, 0.8), fl("steps", 8.0, 2.0, 256.0, 0.05)] },
            Vignette => const {
                &[
                    col("Image", 0.8, 0.8, 0.8),
                    fl("amount", 0.5, 0.0, 1.0, 0.005),
                    fl("size", 0.6, 0.0, 2.0, 0.005),
                    fl("softness", 0.6, 0.01, 2.0, 0.005),
                    fl("roundness", 1.0, 0.0, 1.0, 0.005),
                    col("Color", 0.0, 0.0, 0.0),
                ]
            },
            FilmGrain => const {
                &[
                    col("Image", 0.8, 0.8, 0.8),
                    fl("amount", 0.05, 0.0, 1.0, 0.001),
                    fl("size (px)", 1.0, 0.5, 16.0, 0.01),
                    ch("animated", 1.0, OFF_ON),
                    ch("colored", 0.0, OFF_ON),
                ]
            },
            LensDistortion => const {
                &[col("Image", 0.8, 0.8, 0.8), fl("distortion", 0.0, -1.0, 1.0, 0.002), fl("dispersion", 0.0, 0.0, 1.0, 0.001)]
            },
            Pixelate => const { &[col("Image", 0.8, 0.8, 0.8), fl("size (px)", 8.0, 1.0, 512.0, 0.05)] },
            Sharpen => const {
                &[col("Image", 0.8, 0.8, 0.8), fl("amount", 0.5, 0.0, 8.0, 0.005), fl("radius (px)", 1.0, 0.5, 8.0, 0.005)]
            },
            Kaleidoscope => const {
                &[
                    col("Image", 0.8, 0.8, 0.8),
                    fl("segments", 6.0, 1.0, 64.0, 0.02),
                    fl("rotation°", 0.0, -360.0, 360.0, 0.25),
                    fl("zoom", 1.0, 0.05, 20.0, 0.005),
                ]
            },
            Blur => const { &[col("Image", 0.8, 0.8, 0.8), fl("radius (%)", 1.0, 0.0, 50.0, 0.01)] },
            Bloom => const {
                &[
                    col("Image", 0.8, 0.8, 0.8),
                    fl("intensity", 0.6, 0.0, 16.0, 0.005),
                    fl("threshold", 1.0, 0.0, 100.0, 0.005),
                    fl("soft knee", 0.5, 0.0, 1.0, 0.005),
                    fl("radius", 1.0, 0.5, 4.0, 0.005),
                ]
            },
            MotionBlur => const { &[col("Image", 0.8, 0.8, 0.8), fl("shutter", 0.5, 0.0, 1.0, 0.005)] },
        }
    }
}

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub pos: [f32; 2],
    /// Inline input values, `InputDef::width()` floats per input, in input order.
    pub values: Vec<f32>,
    /// Bypassed (Blender's mute): a formula node passes its input State through.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub muted: bool,
    /// Can't be moved, edited or deleted.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub locked: bool,
    /// Keyframed inputs. Each frame the sampled values are written into `values`
    /// (`Graph::animate`), so animation never needs a shader recompile.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anim: Vec<Track>,
    /// Image file of nodes that read one (`NodeKind::has_file`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

/// Keyframes of one node input (all of its components), e.g. a Mandelbulb's power.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Track {
    pub input: usize,
    /// Sorted by time.
    pub keys: Vec<ValueKey>,
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct ValueKey {
    pub time: f32,
    /// The input's components (a float uses only the first).
    pub value: [f32; 3],
    /// Interpolation towards the next key.
    #[serde(default)]
    pub interp: Interp,
}

impl Track {
    pub fn sample(&self, t: f32) -> Option<[f32; 3]> {
        let curve: Vec<crate::anim::Key<3>> = self.keys.iter().map(|k| (k.time, k.value, k.interp)).collect();
        crate::anim::sample(&curve, t)
    }

    pub fn key_at(&self, t: f32) -> Option<usize> {
        self.keys.iter().position(|k| (k.time - t).abs() < KEY_EPSILON)
    }
}

/// How an input relates to its keyframes at the current time (Blender's field colors).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyState {
    /// Not animated.
    None,
    /// Animated, no key at this time.
    Animated,
    /// A key at this time.
    OnKey,
    /// Animated, but the value was edited and differs from the curve (not keyed yet).
    Changed,
}

impl Node {
    /// Offset of input `j`'s inline value within `values`.
    pub fn value_offset(&self, j: usize) -> usize {
        self.kind.inputs()[..j].iter().map(InputDef::width).sum()
    }

    pub fn value(&self, j: usize) -> f32 {
        self.values[self.value_offset(j)]
    }

    pub fn set(&mut self, j: usize, v: &[f32]) {
        let o = self.value_offset(j);
        self.values[o..o + v.len()].copy_from_slice(v);
    }

    /// Enum inputs are stored as floats; read them back as an index.
    pub fn choice(&self, j: usize) -> usize {
        self.value(j).max(0.0) as usize
    }

    /// Input `j` can be keyframed (see `InputDef::keyable`).
    pub fn keyable(&self, j: usize) -> bool {
        self.kind.inputs().get(j).is_some_and(InputDef::keyable)
    }

    /// Inline values of a fresh node of `kind`.
    pub fn default_values(kind: NodeKind) -> Vec<f32> {
        kind.inputs().iter().flat_map(InputDef::default_values).collect()
    }

    pub fn track(&self, j: usize) -> Option<&Track> {
        self.anim.iter().find(|tr| tr.input == j)
    }

    fn current(&self, j: usize) -> [f32; 3] {
        let (o, w) = (self.value_offset(j), self.kind.inputs()[j].width());
        let mut v = [0.0; 3];
        v[..w].copy_from_slice(&self.values[o..o + w]);
        v
    }

    /// Writes every track's value at `t` into `values` (this node's inline values).
    fn apply_anim(&self, values: &mut [f32], t: f32) {
        for tr in &self.anim {
            if let Some(v) = tr.sample(t) {
                let (o, w) = (self.value_offset(tr.input), self.kind.inputs()[tr.input].width());
                values[o..o + w].copy_from_slice(&v[..w]);
            }
        }
    }

    pub fn key_state(&self, j: usize, t: f32) -> KeyState {
        let Some(tr) = self.track(j) else { return KeyState::None };
        let w = self.kind.inputs()[j].width();
        let (cur, curve) = (self.current(j), tr.sample(t).unwrap_or_default());
        if (0..w).any(|c| (cur[c] - curve[c]).abs() > 1e-6 * (1.0 + curve[c].abs())) {
            KeyState::Changed
        } else if tr.key_at(t).is_some() {
            KeyState::OnKey
        } else {
            KeyState::Animated
        }
    }

    /// Keys input `j`'s current value at `t` (updating a key already there).
    pub fn insert_key(&mut self, j: usize, t: f32) {
        if !self.keyable(j) {
            return;
        }
        let value = self.current(j);
        let tr = match self.anim.iter().position(|tr| tr.input == j) {
            Some(i) => &mut self.anim[i],
            None => {
                self.anim.push(Track { input: j, keys: Vec::new() });
                self.anim.last_mut().unwrap()
            }
        };
        match tr.key_at(t) {
            Some(i) => tr.keys[i].value = value,
            None => {
                tr.keys.push(ValueKey { time: t, value, interp: Interp::Smooth });
                tr.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
            }
        }
    }

    /// Removes input `j`'s key at `t`; the track goes when its last key does.
    pub fn delete_key(&mut self, j: usize, t: f32) -> bool {
        let Some(tr) = self.anim.iter_mut().find(|tr| tr.input == j) else { return false };
        let Some(i) = tr.key_at(t) else { return false };
        tr.keys.remove(i);
        self.anim.retain(|tr| !tr.keys.is_empty());
        true
    }

    /// Removes all of input `j`'s keys; the value stays where it is.
    pub fn clear_track(&mut self, j: usize) -> bool {
        let before = self.anim.len();
        self.anim.retain(|tr| tr.input != j);
        self.anim.len() != before
    }

    /// Equal apart from the values of animated inputs, which follow the playhead.
    fn same_ignoring_animation(&self, o: &Node) -> bool {
        if (self.id, self.kind, self.pos, self.muted, self.locked) != (o.id, o.kind, o.pos, o.muted, o.locked)
            || self.anim != o.anim
            || self.file != o.file
            || self.values.len() != o.values.len()
        {
            return false;
        }
        let animated = |k: usize| {
            self.anim.iter().any(|tr| {
                let off = self.value_offset(tr.input);
                (off..off + self.kind.inputs()[tr.input].width()).contains(&k)
            })
        };
        self.values.iter().zip(&o.values).enumerate().all(|(k, (a, b))| a == b || animated(k))
    }
}

/// Copied nodes (system clipboard as JSON, so it also works between windows).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Clip {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
}

impl Clip {
    const TAG: &str = "fractalful-nodes:";

    pub fn to_text(&self) -> String {
        format!("{}{}", Self::TAG, serde_json::to_string(self).expect("clip serializes"))
    }

    /// `None` when the text isn't copied nodes (any other clipboard content).
    pub fn from_text(text: &str) -> Option<Clip> {
        serde_json::from_str(text.trim().strip_prefix(Self::TAG)?).ok()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Link {
    pub from: NodeId,
    pub from_out: usize,
    pub to: NodeId,
    pub input: usize,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    next_id: NodeId,
    /// Bumped on any change that requires regenerating the shader.
    #[serde(skip)]
    pub structure_rev: u64,
    /// Starting camera (position, yaw, pitch) chosen by a preset, e.g. a spot inside a
    /// space-filling fractal that isn't embedded in its surface.
    pub camera_hint: Option<([f32; 3], f32, f32)>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), links: Vec::new(), next_id: 0, structure_rev: 0, camera_hint: None }
    }

    pub fn add(&mut self, kind: NodeKind, pos: [f32; 2]) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let values = Node::default_values(kind);
        self.nodes.push(Node { id, kind, pos, values, muted: false, locked: false, anim: Vec::new(), file: None });
        self.structure_rev += 1;
        id
    }

    pub fn remove(&mut self, id: NodeId) {
        self.nodes.retain(|n| n.id != id);
        self.links.retain(|l| l.from != id && l.to != id);
        self.structure_rev += 1;
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// The (node, output index) linked into `to`'s input, if any.
    pub fn source(&self, to: NodeId, input: usize) -> Option<(NodeId, usize)> {
        self.links.iter().find(|l| l.to == to && l.input == input).map(|l| (l.from, l.from_out))
    }

    pub fn find(&self, kind: NodeKind) -> Option<&Node> {
        self.nodes.iter().find(|n| n.kind == kind)
    }

    /// Links `from`'s output `from_out` into `to`'s input, replacing any existing link
    /// there. Refuses (returning false) type mismatches, enum inputs and cycles.
    pub fn connect(&mut self, from: NodeId, from_out: usize, to: NodeId, input: usize) -> bool {
        let (Some(a), Some(b)) = (self.node(from), self.node(to)) else { return false };
        let (Some(o), Some(i)) = (a.kind.outputs().get(from_out), b.kind.inputs().get(input)) else { return false };
        if !i.linkable() || !o.ty.compatible(i.ty) || from == to || self.reaches(to, from) {
            return false;
        }
        self.links.retain(|l| !(l.to == to && l.input == input));
        self.links.push(Link { from, from_out, to, input });
        self.structure_rev += 1;
        true
    }

    pub fn disconnect(&mut self, to: NodeId, input: usize) -> Option<(NodeId, usize)> {
        let src = self.source(to, input)?;
        self.links.retain(|l| !(l.to == to && l.input == input));
        self.structure_rev += 1;
        Some(src)
    }

    /// Unlinks every input of the output nodes, leaving the rest of the graph intact.
    /// What's left compiles to the trivial fractal (untouched domain, identity step,
    /// built-in material): the safe state to reopen a graph that hung the app.
    /// Returns the number of links removed.
    pub fn cut_outputs(&mut self) -> usize {
        let outputs: Vec<NodeId> =
            self.nodes.iter().filter(|n| n.kind.category() == Category::Output).map(|n| n.id).collect();
        let before = self.links.len();
        self.links.retain(|l| !outputs.contains(&l.to));
        let cut = before - self.links.len();
        self.structure_rev += (cut > 0) as u64;
        cut
    }

    /// True if data flows from `a` to `b` (i.e. `b` depends on `a`).
    fn reaches(&self, a: NodeId, b: NodeId) -> bool {
        let mut stack = vec![a];
        let mut seen = Vec::new();
        while let Some(n) = stack.pop() {
            if n == b {
                return true;
            }
            if !seen.contains(&n) {
                seen.push(n);
                stack.extend(self.links.iter().filter(|l| l.from == n).map(|l| l.to));
            }
        }
        false
    }

    /// Iteration Output settings: (bailout, bound radius, linear DE).
    /// Iteration Output settings: (bailout, bound radius, distance mode index).
    pub fn output_settings(&self) -> (f32, f32, u32) {
        self.find(NodeKind::IterOutput)
            .map_or((4.0, 1.5, 0), |o| (o.value(OUT_BAILOUT), o.value(OUT_BOUND), o.choice(OUT_DE_MODE) as u32))
    }

    /// Nodes that feed the given output node, directly or indirectly.
    pub fn upstream(&self, root: NodeId) -> Vec<NodeId> {
        let mut stack = vec![root];
        let mut seen = Vec::new();
        while let Some(n) = stack.pop() {
            if !seen.contains(&n) {
                seen.push(n);
                stack.extend(self.links.iter().filter(|l| l.to == n).map(|l| l.from));
            }
        }
        seen
    }

    /// Nodes reachable from any output node (disconnected nodes don't count).
    fn live_nodes(&self) -> Vec<NodeId> {
        let roots = self.nodes.iter().filter(|n| n.kind.outputs().is_empty()).map(|n| n.id);
        roots.flat_map(|r| self.upstream(r)).collect()
    }

    /// True if a connected driver node depends on time, so the image changes every frame.
    pub fn uses_time(&self) -> bool {
        self.live_nodes()
            .iter()
            .any(|&id| {
                self.node(id).is_some_and(|n| n.kind.category() == Category::Driver || n.kind.is_live() || !n.anim.is_empty())
            })
    }

    /// `Some(view distance)` when the domain graph repeats space, making the scene
    /// unbounded (the bounding sphere is replaced by a view distance).
    ///
    /// Kleinian / planar distance modes are space-filling too: their view distance is
    /// the Domain Output's if present, else 10x the bound radius (which then acts as
    /// the feature size for fog and shadows).
    pub fn unbounded_view_distance(&self) -> Option<f32> {
        let domain = self.find(NodeKind::DomainOutput);
        let (_, bound, mode) = self.output_settings();
        if mode == DE_KLEINIAN || mode == DE_PLANAR {
            return Some(domain.map_or(bound * 10.0, |d| d.value(1)));
        }
        let out = domain?;
        let repeats = self.upstream(out.id).iter().any(|&id| self.node(id).is_some_and(|n| n.kind == NodeKind::Repeat));
        repeats.then(|| out.value(1))
    }

    /// A starting camera position: in front of a bounded fractal, or for a repeating
    /// field in the gap between cells, looking down the corridor along +z.
    /// Returns (position, yaw, pitch); presets can override it with `camera_hint`.
    pub fn suggested_camera(&self) -> ([f32; 3], f32, f32) {
        if let Some(hint) = self.camera_hint {
            return hint;
        }
        let bound = self.output_settings().1;
        match self.find(NodeKind::Repeat).filter(|_| self.unbounded_view_distance().is_some()) {
            Some(rep) => {
                let o = rep.value_offset(1);
                let cell = &rep.values[o..o + 3];
                ([cell[0] * 0.5, cell[1] * 0.5, -cell[2] * 2.5], 0.0, 0.0)
            }
            None => ([0.0, 0.0, -bound * 1.7], 0.0, 0.0),
        }
    }

    /// Whether any surface can be smooth enough for ray-traced reflections (roughness
    /// below `REFLECTION_MAX_ROUGHNESS`). When not, codegen compiles the reflection code
    /// out entirely: just having it in the shader costs registers (about 5-10%).
    pub fn may_reflect(&self) -> bool {
        self.find(NodeKind::ColorOutput).is_some_and(|m| {
            self.source(m.id, MAT_ROUGHNESS).is_some() || m.value(MAT_ROUGHNESS) < REFLECTION_MAX_ROUGHNESS
        })
    }

    /// Same nodes, links, values and keys (ignores `structure_rev`); used by undo
    /// history. Values of animated inputs are ignored: they follow the playhead.
    pub fn same_content(&self, other: &Graph) -> bool {
        self.links == other.links
            && self.camera_hint == other.camera_hint
            && self.nodes.len() == other.nodes.len()
            && self.nodes.iter().zip(&other.nodes).all(|(a, b)| a.same_ignoring_animation(b))
    }

    /// Writes every keyframed input's value at time `t` into the graph.
    pub fn animate(&mut self, t: f32) {
        for n in &mut self.nodes {
            if !n.anim.is_empty() {
                let mut values = std::mem::take(&mut n.values);
                n.apply_anim(&mut values, t);
                n.values = values;
            }
        }
    }

    /// `packed_values` with the animation at `t` applied, leaving the graph untouched
    /// (video export samples several instants per frame for motion blur).
    pub fn packed_at(&self, t: f32) -> Vec<f32> {
        let mut out = Vec::new();
        for n in &self.nodes {
            let start = out.len();
            out.extend_from_slice(&n.values);
            n.apply_anim(&mut out[start..], t);
        }
        out
    }

    pub fn is_animated(&self) -> bool {
        self.nodes.iter().any(|n| !n.anim.is_empty())
    }

    /// Changes whenever any key changes (time, value, interpolation, added / removed).
    pub fn anim_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for n in self.nodes.iter().filter(|n| !n.anim.is_empty()) {
            n.id.hash(&mut h);
            for tr in &n.anim {
                tr.input.hash(&mut h);
                for k in &tr.keys {
                    (k.time.to_bits(), k.value.map(f32::to_bits), k.interp as u8).hash(&mut h);
                }
            }
        }
        h.finish()
    }

    /// Times that have at least one node key, sorted, with near-equal times merged.
    pub fn key_times(&self) -> Vec<f32> {
        let mut times: Vec<f32> = self.nodes.iter().flat_map(|n| &n.anim).flat_map(|tr| tr.keys.iter().map(|k| k.time)).collect();
        times.sort_by(f32::total_cmp);
        times.dedup_by(|b, a| (*b - *a).abs() < KEY_EPSILON);
        times
    }

    fn keys_mut(&mut self) -> impl Iterator<Item = &mut ValueKey> {
        self.nodes.iter_mut().flat_map(|n| &mut n.anim).flat_map(|tr| &mut tr.keys)
    }

    /// Moves every key at one of `times` by `delta`. Where a moved key lands on another
    /// key of the same input, the moved one wins.
    pub fn move_keys(&mut self, times: &[f32], delta: f32) {
        let hit = |t: f32| times.iter().any(|&s| (s - t).abs() < KEY_EPSILON);
        for n in &mut self.nodes {
            for tr in &mut n.anim {
                let mut moved = Vec::new();
                tr.keys.retain(|k| {
                    if hit(k.time) {
                        moved.push(ValueKey { time: (k.time + delta).max(0.0), ..*k });
                        false
                    } else {
                        true
                    }
                });
                tr.keys.retain(|k| !moved.iter().any(|m| (m.time - k.time).abs() < KEY_EPSILON));
                tr.keys.extend(moved);
                tr.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
            }
        }
    }

    /// Deletes every node key at one of `times`. Returns how many went.
    pub fn delete_keys_at(&mut self, times: &[f32]) -> usize {
        let mut n_deleted = 0;
        for n in &mut self.nodes {
            for tr in &mut n.anim {
                let before = tr.keys.len();
                tr.keys.retain(|k| !times.iter().any(|&s| (s - k.time).abs() < KEY_EPSILON));
                n_deleted += before - tr.keys.len();
            }
            n.anim.retain(|tr| !tr.keys.is_empty());
        }
        n_deleted
    }

    pub fn set_interp_at(&mut self, times: &[f32], interp: Interp) {
        for k in self.keys_mut().filter(|k| times.iter().any(|&s| (s - k.time).abs() < KEY_EPSILON)) {
            k.interp = interp;
        }
    }

    /// Auto-key: every animated input whose value was edited away from its curve gets a
    /// key at `t`. Returns true if anything was keyed.
    pub fn autokey(&mut self, t: f32) -> bool {
        let mut keyed = false;
        for n in &mut self.nodes {
            let inputs: Vec<usize> = n.anim.iter().map(|tr| tr.input).collect();
            for j in inputs {
                if n.key_state(j, t) == KeyState::Changed {
                    n.insert_key(j, t);
                    keyed = true;
                }
            }
        }
        keyed
    }

    /// Same nodes, links and enum choices: the compiled shader would be identical
    /// (only inline values, node positions or the camera hint may differ).
    pub fn same_structure(&self, other: &Graph) -> bool {
        self.links == other.links
            && self.nodes.len() == other.nodes.len()
            && self.nodes.iter().zip(&other.nodes).all(|(a, b)| {
                a.id == b.id
                    && a.kind == b.kind
                    && a.muted == b.muted
                    && (a.kind.is_live()
                        || a.kind.inputs().iter().enumerate().all(|(j, d)| d.choices.is_empty() || a.value(j) == b.value(j)))
            })
    }

    /// Makes a graph read from disk safe to compile: resizes inline values to the
    /// current node definitions (new inputs get their defaults) and drops links whose
    /// sockets no longer exist or no longer type-check.
    pub fn repair(&mut self) {
        for n in &mut self.nodes {
            let defaults = Node::default_values(n.kind);
            if n.values.len() != defaults.len() {
                let keep = n.values.len().min(defaults.len());
                n.values = n.values[..keep].iter().copied().chain(defaults[keep..].iter().copied()).collect();
            }
        }
        for n in &mut self.nodes {
            let kind = n.kind;
            n.anim.retain(|tr| kind.inputs().get(tr.input).is_some_and(InputDef::keyable) && !tr.keys.is_empty());
            for tr in &mut n.anim {
                tr.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
            }
        }
        let links = std::mem::take(&mut self.links);
        for l in links {
            self.connect(l.from, l.from_out, l.to, l.input);
        }
        self.next_id = self.nodes.iter().map(|n| n.id + 1).max().unwrap_or(0).max(self.next_id);
    }

    /// Live nodes (audio, Wiggle) and the index of their first output in the live slot array;
    /// output k of the node is at slot `base + k`. Fails when they don't fit.
    pub fn live_slots(&self) -> Result<Vec<(NodeId, usize)>, String> {
        let mut next = 0;
        let mut slots = Vec::new();
        for n in self.nodes.iter().filter(|n| n.kind.is_live()) {
            slots.push((n.id, next));
            next += n.kind.outputs().len();
        }
        if next > LIVE_SLOTS {
            return Err(format!("too many audio / Wiggle nodes ({next} outputs > {LIVE_SLOTS})"));
        }
        Ok(slots)
    }

    /// The nodes `ids` and the links between them, for the clipboard. Output nodes are
    /// left out: a graph can only have one of each.
    pub fn copy(&self, ids: &[NodeId]) -> Clip {
        let nodes: Vec<Node> = self
            .nodes
            .iter()
            .filter(|n| ids.contains(&n.id) && n.kind.category() != Category::Output)
            .cloned()
            .collect();
        let has = |id: NodeId| nodes.iter().any(|n| n.id == id);
        let links = self.links.iter().filter(|l| has(l.from) && has(l.to)).copied().collect();
        Clip { nodes, links }
    }

    /// Adds a copy of `clip` with fresh ids, its top-left node at `at` (unlocked, so it
    /// can be moved into place). Nodes that can't go into a `kind` graph, or unique ones
    /// it already has, are skipped. Returns the new nodes' ids.
    pub fn paste(&mut self, clip: &Clip, at: [f32; 2], kind: GraphKind) -> Vec<NodeId> {
        let min = clip.nodes.iter().fold([f32::INFINITY; 2], |m, n| [m[0].min(n.pos[0]), m[1].min(n.pos[1])]);
        let mut map = Vec::new();
        for n in &clip.nodes {
            if !n.kind.allowed_in(kind) || (n.kind.unique() && self.find(n.kind).is_some()) {
                continue;
            }
            // Values carry over only when the node layout matches (it always does within a version).
            let id = self.add(n.kind, [at[0] + n.pos[0] - min[0], at[1] + n.pos[1] - min[1]]);
            let node = self.node_mut(id).unwrap();
            if n.values.len() == node.values.len() {
                node.values.clone_from(&n.values);
            }
            node.muted = n.muted && n.kind.mutable();
            node.file.clone_from(&n.file);
            if n.values.len() == node.values.len() {
                node.anim = n.anim.iter().filter(|tr| n.keyable(tr.input)).cloned().collect();
            }
            map.push((n.id, id));
        }
        let new = |old: NodeId| map.iter().find(|m| m.0 == old).map(|m| m.1);
        for l in &clip.links {
            if let (Some(from), Some(to)) = (new(l.from), new(l.to)) {
                self.connect(from, l.from_out, to, l.input);
            }
        }
        map.into_iter().map(|m| m.1).collect()
    }

    /// Inline values packed in node order; `codegen` addresses them the same way.
    pub fn packed_values(&self) -> Vec<f32> {
        self.nodes.iter().flat_map(|n| n.values.iter().copied()).collect()
    }
}

impl Default for Graph {
    fn default() -> Self {
        preset("Mandelbulb").0
    }
}

/// The look set by render settings before the world and processing graphs existed.
/// Old projects get graphs built from it, so they look the same as before.
#[derive(Clone, Debug)]
pub struct LegacyLook {
    pub sky_top: [f32; 3],
    pub sky_bottom: [f32; 3],
    pub sky_solid: bool,
    /// Light direction (radians).
    pub light_yaw: f32,
    pub light_pitch: f32,
    pub light_intensity: f32,
    pub fog: f32,
    pub bloom_enabled: bool,
    /// Bloom intensity, threshold, soft knee, radius.
    pub bloom: [f32; 4],
    /// Video export motion blur.
    pub shutter: f32,
}

impl Default for LegacyLook {
    fn default() -> Self {
        Self {
            sky_top: [0.10, 0.13, 0.20],
            sky_bottom: [0.02, 0.02, 0.035],
            sky_solid: false,
            light_yaw: 0.8,
            light_pitch: 0.7,
            light_intensity: 1.0,
            fog: 0.02,
            bloom_enabled: true,
            bloom: [0.6, 1.0, 0.5, 1.0],
            shutter: 0.5,
        }
    }
}

impl Graph {
    /// A world graph reproducing the old background gradient, light and fog. The sun
    /// disc stays hidden (size 0), as the old light had none.
    pub fn world_from(l: &LegacyLook) -> Graph {
        let mut g = Graph::new();
        let sky = g.add(NodeKind::SkyGradient, [40.0, 40.0]);
        let out = g.add(NodeKind::WorldOutput, [340.0, 40.0]);
        let bottom = if l.sky_solid { l.sky_top } else { l.sky_bottom };
        let horizon: Vec<f32> = l.sky_top.iter().zip(bottom).map(|(t, b)| 0.5 * (t + b)).collect();
        let n = g.node_mut(sky).unwrap();
        n.set(1, &l.sky_top);
        n.set(2, &horizon);
        n.set(3, &bottom);
        let n = g.node_mut(out).unwrap();
        n.set(WORLD_SUN_AZIMUTH, &[l.light_yaw.to_degrees()]);
        n.set(WORLD_SUN_ELEVATION, &[l.light_pitch.to_degrees()]);
        n.set(WORLD_SUN_STRENGTH, &[1.3 * l.light_intensity]);
        n.set(WORLD_SUN_SIZE, &[0.0]);
        n.set(WORLD_FOG, &[l.fog]);
        g.connect(sky, 0, out, WORLD_COLOR);
        g
    }

    /// A processing graph with the old bloom (muted when it was off) and, if the video
    /// export had a shutter, a Motion Blur node.
    pub fn post_from(l: &LegacyLook) -> Graph {
        let mut g = Graph::new();
        let render = g.add(NodeKind::RenderLayers, [40.0, 40.0]);
        let bloom = g.add(NodeKind::Bloom, [300.0, 40.0]);
        let out = g.add(NodeKind::PostOutput, [620.0, 40.0]);
        let n = g.node_mut(bloom).unwrap();
        n.set(1, &l.bloom[..1]);
        n.set(2, &l.bloom[1..2]);
        n.set(3, &l.bloom[2..3]);
        n.set(4, &l.bloom[3..4]);
        n.muted = !l.bloom_enabled;
        g.connect(render, 0, bloom, 0);
        if l.shutter > 0.0 {
            let mb = g.add(NodeKind::MotionBlur, [620.0, 40.0]);
            g.node_mut(mb).unwrap().set(1, &[l.shutter]);
            g.node_mut(out).unwrap().pos = [880.0, 40.0];
            g.connect(bloom, 0, mb, 0);
            g.connect(mb, 0, out, 0);
        } else {
            g.connect(bloom, 0, out, 0);
        }
        g
    }

    /// Processing graph: the shutter of a Motion Blur node feeding the output (0 = none).
    pub fn motion_blur_shutter(&self) -> f32 {
        let Some(out) = self.find(NodeKind::PostOutput) else { return 0.0 };
        let up = self.upstream(out.id);
        self.nodes
            .iter()
            .find(|n| n.kind == NodeKind::MotionBlur && !n.muted && up.contains(&n.id))
            .map_or(0.0, |n| n.value(1).clamp(0.0, 1.0))
    }

    /// Processing graph: the Image Output's view transform index.
    pub fn view_transform(&self) -> usize {
        self.find(NodeKind::PostOutput).map_or(0, |n| n.choice(POST_VIEW_TRANSFORM))
    }
}

/// A scene's three graphs.
#[derive(Clone)]
pub struct Graphs {
    pub material: Graph,
    pub world: Graph,
    pub post: Graph,
}

impl Default for Graphs {
    fn default() -> Self {
        let look = LegacyLook::default();
        Self { material: Graph::default(), world: Graph::world_from(&look), post: Graph::post_from(&look) }
    }
}

/// A node key track snapshot: (graph, node, tracks), see `Graphs::anim_snapshot`.
pub type AnimSnapshot = Vec<(GraphKind, NodeId, Vec<Track>)>;

impl Graphs {
    pub fn get(&self, k: GraphKind) -> &Graph {
        match k {
            GraphKind::Material => &self.material,
            GraphKind::World => &self.world,
            GraphKind::Post => &self.post,
        }
    }

    pub fn get_mut(&mut self, k: GraphKind) -> &mut Graph {
        match k {
            GraphKind::Material => &mut self.material,
            GraphKind::World => &mut self.world,
            GraphKind::Post => &mut self.post,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Graph> {
        [&self.material, &self.world, &self.post].into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Graph> {
        [&mut self.material, &mut self.world, &mut self.post].into_iter()
    }

    pub fn same_content(&self, o: &Graphs) -> bool {
        self.iter().zip(o.iter()).all(|(a, b)| a.same_content(b))
    }

    pub fn animate(&mut self, t: f32) {
        self.iter_mut().for_each(|g| g.animate(t));
    }

    pub fn is_animated(&self) -> bool {
        self.iter().any(Graph::is_animated)
    }

    pub fn anim_fingerprint(&self) -> u64 {
        self.iter().fold(0u64, |h, g| h.rotate_left(17) ^ g.anim_fingerprint())
    }

    /// Times with at least one node key in any graph, sorted, near-equal times merged.
    pub fn key_times(&self) -> Vec<f32> {
        let mut times: Vec<f32> = self.iter().flat_map(Graph::key_times).collect();
        times.sort_by(f32::total_cmp);
        times.dedup_by(|b, a| (*b - *a).abs() < KEY_EPSILON);
        times
    }

    pub fn move_keys(&mut self, times: &[f32], delta: f32) {
        self.iter_mut().for_each(|g| g.move_keys(times, delta));
    }

    pub fn delete_keys_at(&mut self, times: &[f32]) -> usize {
        self.iter_mut().map(|g| g.delete_keys_at(times)).sum()
    }

    pub fn set_interp_at(&mut self, times: &[f32], interp: Interp) {
        self.iter_mut().for_each(|g| g.set_interp_at(times, interp));
    }

    pub fn autokey(&mut self, t: f32) -> bool {
        self.iter_mut().fold(false, |keyed, g| g.autokey(t) | keyed)
    }

    /// Every animated node's tracks, to restore while dragging keys in the timeline.
    pub fn anim_snapshot(&self) -> AnimSnapshot {
        GraphKind::ALL
            .iter()
            .flat_map(|&k| self.get(k).nodes.iter().filter(|n| !n.anim.is_empty()).map(move |n| (k, n.id, n.anim.clone())))
            .collect()
    }

    pub fn restore_anim(&mut self, snapshot: &AnimSnapshot) {
        for (k, id, anim) in snapshot {
            if let Some(n) = self.get_mut(*k).node_mut(*id) {
                n.anim.clone_from(anim);
            }
        }
    }
}

/// Built-in presets by category (the Load preset menu). Every name is handled by `preset`.
pub const PRESET_CATEGORIES: &[(&str, &[&str])] = &[
    (
        "Classic",
        &[
            "Mandelbulb",
            "Mandelbox",
            "Menger sponge",
            "Bulb + Box hybrid",
            "Amazing Surf",
            "Kaliset",
            "Kaleidoscopic IFS",
            "Icosahedral IFS",
            "Twisted Sierpinski",
            "Burning Ship bulb",
            "Pine Tree",
            "Quick Dudley",
            "Snowflake Mandelbox",
        ],
    ),
    ("Materials", &["Glowing bulb", "Chrome Mandelbox", "Jade bulb", "Lava bulb"]),
    (
        "Animated",
        &[
            "Morphing bulb",
            "Breathing bulb",
            "Morphing Mandelbox",
            "Psychedelic Juliabulb",
            "Kaleidoscope",
            "Xenodream drift",
            "Beat-synced bulb",
        ],
    ),
    (
        "Infinite spaces",
        &[
            "Infinite Menger field",
            "Spinning bulb grid",
            "Tumbling bulb field",
            "Infinite fly-through",
            "Kleinian dreams",
            "Apollonian cathedral",
        ],
    ),
    (
        "Landscapes",
        &[
            "Menger canyons",
            "Molten canyons",
            "Frozen catacombs",
            "Menger corridor",
            "Sierpinski lattice",
            "Pagoda fields",
            "Icosahedral glacier",
            "Mandelbox peaks",
            "Bulb garden",
            "Kleinian badlands",
            "Kleinian lagoon",
        ],
    ),
];

/// Every preset name, category by category.
pub fn presets() -> impl Iterator<Item = &'static str> {
    PRESET_CATEGORIES.iter().flat_map(|(_, names)| names.iter().copied())
}

/// Built-in starting points. Returns the graph and a suitable iteration count.
pub fn preset(name: &str) -> (Graph, u32) {
    use NodeKind::*;
    let mut g = Graph::new();
    let start = g.add(IterStart, [20.0, 120.0]);
    let output = g.add(IterOutput, [760.0, 60.0]);
    let set_out_mode = |g: &mut Graph, bailout: f32, bound: f32, mode: u32| {
        let o = g.node_mut(output).unwrap();
        o.set(OUT_BAILOUT, &[bailout]);
        o.set(OUT_BOUND, &[bound]);
        o.set(OUT_DE_MODE, &[mode as f32]);
    };
    let set_out = |g: &mut Graph, bailout: f32, bound: f32, linear: bool| set_out_mode(g, bailout, bound, linear as u32);
    // A single formula node between start and output.
    let single = |g: &mut Graph, kind: NodeKind| {
        let n = g.add(kind, [300.0, 60.0]);
        g.connect(start, 0, n, 0);
        g.connect(n, 0, output, 0);
        n
    };

    let iters = match name {
        "Mandelbox" => {
            let b = single(&mut g, Mandelbox);
            g.node_mut(b).unwrap().set(1, &[-1.5]);
            set_out(&mut g, 1024.0, 4.5, true);
            16
        }
        "Menger sponge" => {
            single(&mut g, Menger);
            set_out(&mut g, 1024.0, 2.0, true);
            10
        }
        "Bulb + Box hybrid" => {
            let bulb = g.add(Mandelbulb, [280.0, 0.0]);
            let bx = g.add(Mandelbox, [280.0, 130.0]);
            let alt = g.add(Alternate, [530.0, 60.0]);
            g.node_mut(bx).unwrap().set(1, &[-1.5]);
            g.connect(start, 0, bulb, 0);
            g.connect(start, 0, bx, 0);
            g.connect(bulb, 0, alt, 0);
            g.connect(bx, 0, alt, 1);
            g.connect(alt, 0, output, 0);
            set_out(&mut g, 16.0, 2.0, false);
            12
        }
        "Amazing Surf" => {
            let s = single(&mut g, AmazingSurf);
            g.node_mut(s).unwrap().set(1, &[-1.6]);
            set_out(&mut g, 1024.0, 4.5, true);
            14
        }
        "Kaliset" => {
            let k = single(&mut g, Kaliset);
            g.node_mut(k).unwrap().set(1, &[1.2]);
            set_out(&mut g, 1024.0, 2.0, true);
            12
        }
        "Kaleidoscopic IFS" => {
            single(&mut g, Kifs);
            set_out(&mut g, 1024.0, 1.5, true);
            12
        }
        "Morphing bulb" => {
            // Power drops from 8 to 3 over the iterations: Iteration Info -> Map Range -> power.
            let info = g.add(IterInfo, [20.0, 200.0]);
            let map = g.add(MapRange, [260.0, 150.0]);
            let bulb = g.add(Mandelbulb, [520.0, 40.0]);
            let m = g.node_mut(map).unwrap();
            m.set(2, &[8.0]);
            m.set(3, &[8.0]);
            m.set(4, &[3.0]);
            g.connect(info, 0, map, 0);
            g.connect(start, 0, bulb, 0);
            g.connect(map, 0, bulb, 1);
            g.connect(bulb, 0, output, 0);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Twisted Sierpinski" => {
            let tw = g.add(Twist, [280.0, 60.0]);
            let s = g.add(Sierpinski, [520.0, 60.0]);
            g.connect(start, 0, tw, 0);
            g.connect(tw, 0, s, 0);
            g.connect(s, 0, output, 0);
            set_out(&mut g, 1024.0, 2.0, true);
            14
        }
        "Beat-synced bulb" => {
            // Each kick punches the power up (Kick pulse 0..1 -> power 8..9.5); each beat
            // turns the whole bulb 22.5° with an eased snap (Beat glide -> domain rotation).
            let kick = g.add(Beat, [20.0, 220.0]);
            let k = g.node_mut(kick).unwrap();
            k.set(0, &[2.0]);
            k.set(2, &[0.25]);
            let power = g.add(MapRange, [260.0, 220.0]);
            let m = g.node_mut(power).unwrap();
            m.set(3, &[8.0]);
            m.set(4, &[9.5]);
            let bulb = g.add(Mandelbulb, [520.0, 60.0]);
            g.connect(kick, 0, power, 0);
            g.connect(power, 0, bulb, 1);
            g.connect(start, 0, bulb, 0);
            g.connect(bulb, 0, output, 0);

            let beat = g.add(Beat, [20.0, -300.0]);
            g.node_mut(beat).unwrap().set(2, &[0.35]);
            let deg = g.add(Math, [260.0, -300.0]);
            let d = g.node_mut(deg).unwrap();
            d.set(0, &[2.0]);
            d.set(2, &[22.5]);
            let angles = g.add(CombineXyz, [480.0, -300.0]);
            let pos = g.add(Position, [480.0, -140.0]);
            let rot = g.add(RotateVector, [700.0, -260.0]);
            let domain = g.add(DomainOutput, [940.0, -260.0]);
            g.connect(beat, 3, deg, 1);
            g.connect(deg, 0, angles, 1);
            g.connect(pos, 0, rot, 0);
            g.connect(angles, 0, rot, 1);
            g.connect(rot, 0, domain, 0);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Breathing bulb" => {
            // Oscillator (sine, 0.1 Hz) drives the bulb power between 5 and 9.
            let osc = g.add(Oscillator, [20.0, 220.0]);
            let bulb = g.add(Mandelbulb, [300.0, 60.0]);
            let o = g.node_mut(osc).unwrap();
            o.set(1, &[0.1]);
            o.set(2, &[2.0]);
            o.set(3, &[7.0]);
            g.connect(start, 0, bulb, 0);
            g.connect(osc, 0, bulb, 1);
            g.connect(bulb, 0, output, 0);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Infinite Menger field" => {
            single(&mut g, Menger);
            set_out(&mut g, 1024.0, 2.0, true);
            add_repeat(&mut g, 4.0, false);
            8
        }
        "Spinning bulb grid" => {
            single(&mut g, Mandelbulb);
            set_out(&mut g, 4.0, 1.5, false);
            let (rep, dout) = add_repeat(&mut g, 3.0, true);
            // Each cell's bulb spins: Repeat -> Rotate Vector (angle from Time) -> output.
            let time = g.add(Time, [20.0, -160.0]);
            let speed = g.add(Math, [240.0, -200.0]);
            let rot = g.add(RotateVector, [500.0, -260.0]);
            g.node_mut(speed).unwrap().set(0, &[2.0]); // multiply
            g.node_mut(speed).unwrap().set(2, &[20.0]); // degrees per second
            g.connect(time, 0, speed, 1);
            g.connect(rep, 0, rot, 0);
            g.connect(speed, 0, rot, 1);
            g.connect(rot, 0, dout, 0);
            8
        }
        "Burning Ship bulb" => {
            let b = single(&mut g, Mandelbulb);
            g.node_mut(b).unwrap().set(3, &[2.0]);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Pine Tree" => {
            single(&mut g, PineTree);
            set_out(&mut g, 16.0, 2.5, false);
            10
        }
        "Quick Dudley" => {
            single(&mut g, QuickDudley);
            set_out(&mut g, 16.0, 2.5, false);
            12
        }
        "Icosahedral IFS" => {
            let n = single(&mut g, PolyIfs);
            let p = g.node_mut(n).unwrap();
            p.set(2, &[2.2]);
            p.set(3, &[1.0, 0.4, 0.0]);
            p.set(4, &[12.0]);
            set_out(&mut g, 1024.0, 1.5, true);
            12
        }
        "Glowing bulb" | "Jade bulb" => {
            single(&mut g, Mandelbulb);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Chrome Mandelbox" => {
            let b = single(&mut g, Mandelbox);
            g.node_mut(b).unwrap().set(1, &[-1.5]);
            set_out(&mut g, 1024.0, 4.5, true);
            16
        }
        "Kleinian dreams" => {
            // The inversion size breathes slowly, reshaping the whole structure.
            let k = single(&mut g, PseudoKleinian);
            let osc = oscillator(&mut g, [20.0, 220.0], 0.04, 0.05, 1.0);
            g.connect(osc, 0, k, 2);
            set_out_mode(&mut g, 4096.0, 2.0, DE_KLEINIAN);
            // Open space found by sampling the distance field (see eval tests).
            g.camera_hint = Some(([-0.94, -0.94, -0.94], 0.6, 0.35));
            12
        }
        "Apollonian cathedral" => {
            let a = single(&mut g, Apollonian);
            let osc = oscillator(&mut g, [20.0, 220.0], 0.03, 0.04, 1.25);
            g.connect(osc, 0, a, 1);
            set_out_mode(&mut g, 4096.0, 1.5, DE_PLANAR);
            g.camera_hint = Some(([0.94, 1.0, 1.8], 2.4, -0.2));
            8
        }
        "Morphing Mandelbox" => {
            // Box scale sweeps between -2.4 and -1.6.
            let b = single(&mut g, Mandelbox);
            let osc = oscillator(&mut g, [20.0, 220.0], 0.05, 0.4, -2.0);
            g.connect(osc, 0, b, 1);
            set_out(&mut g, 1024.0, 6.0, true);
            g.camera_hint = Some(([0.0, 0.0, -6.5], 0.0, 0.0));
            16
        }
        "Psychedelic Juliabulb" => {
            // Julia bulb: z^8 + k (c weight 0), with k drifting on three oscillators.
            let bulb = g.add(Mandelbulb, [260.0, 0.0]);
            let tr = g.add(Translate, [520.0, 60.0]);
            let comb = g.add(CombineXyz, [260.0, 200.0]);
            g.node_mut(bulb).unwrap().set(2, &[0.0]);
            for (axis, (freq, amp, center)) in [(0.031, 0.25, -0.35), (0.023, 0.25, 0.45), (0.017, 0.2, 0.25)].into_iter().enumerate() {
                let o = oscillator(&mut g, [20.0, 160.0 + 110.0 * axis as f32], freq, amp, center);
                g.connect(o, 0, comb, axis);
            }
            g.connect(start, 0, bulb, 0);
            g.connect(bulb, 0, tr, 0);
            g.connect(comb, 0, tr, 1);
            g.connect(tr, 0, output, 0);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Infinite fly-through" => {
            let n = single(&mut g, PolyIfs);
            let p = g.node_mut(n).unwrap();
            p.set(2, &[2.2]);
            p.set(3, &[1.0, 0.4, 0.0]);
            p.set(4, &[12.0]);
            set_out(&mut g, 1024.0, 1.5, true);
            // Domain: (position + (0, 0, 0.5 t)) -> Repeat. Space scrolls past a static
            // camera, so it flies forever.
            let (rep, _) = add_repeat(&mut g, 2.6, false);
            let pos = g.upstream(rep).into_iter().find(|&id| g.node(id).is_some_and(|n| n.kind == Position)).unwrap();
            let time = g.add(Time, [20.0, -200.0]);
            g.node_mut(time).unwrap().set(0, &[0.5]);
            let comb = g.add(CombineXyz, [240.0, -200.0]);
            let add = g.add(VectorMath, [480.0, -300.0]);
            g.connect(time, 0, comb, 2);
            g.connect(pos, 0, add, 1);
            g.connect(comb, 0, add, 2);
            g.connect(add, 0, rep, 0);
            10
        }
        "Kaleidoscope" => {
            // The fold rotation angle swings +-25 degrees (a full turn would shrink the set).
            let n = single(&mut g, Kifs);
            let osc = oscillator(&mut g, [20.0, 220.0], 0.06, 25.0, 15.0);
            g.connect(osc, 0, n, 3);
            set_out(&mut g, 1024.0, 2.0, true);
            g.camera_hint = Some(([0.0, 0.0, -2.2], 0.0, 0.0));
            12
        }
        "Snowflake Mandelbox" => {
            // Space is folded into six mirrored sectors around z once, before iterating.
            // (A Polar Fold inside the loop would turn far points' z against c and stop
            // them escaping: the set would fill space.)
            let b = single(&mut g, Mandelbox);
            g.node_mut(b).unwrap().set(1, &[-1.5]);
            set_out(&mut g, 1024.0, 4.5, true);
            let pos = g.add(Position, [20.0, -320.0]);
            let polar = g.add(PolarRepeat, [240.0, -340.0]);
            let dout = g.add(DomainOutput, [760.0, -320.0]);
            g.connect(pos, 0, polar, 0);
            g.connect(polar, 0, dout, 0);
            // Oblique, so the hexagonal prism shows depth rather than one flat face.
            g.camera_hint = Some(([4.5, 4.0, -6.0], -0.6435, -0.4900));
            16
        }
        "Xenodream drift" => {
            // The angle offsets wander on two Wiggle drivers: an endlessly morphing bulb.
            let x = single(&mut g, Xenodreambuie);
            for (j, seed) in [(2, 0.0), (3, 1.0)] {
                let w = g.add(Wiggle, [20.0, 120.0 + 130.0 * j as f32]);
                let n = g.node_mut(w).unwrap();
                n.set(0, &[0.04]);
                n.set(1, &[35.0]);
                n.set(4, &[seed]);
                g.connect(w, 0, x, j);
            }
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
        "Tumbling bulb field" => {
            single(&mut g, Mandelbulb);
            set_out(&mut g, 4.0, 1.5, false);
            let (rep, dout) = add_repeat(&mut g, 3.0, false);
            // Each cell gets its own orientation (Cell -> White Noise -> angles) and
            // tumbles slowly from there (+ Time * 12 degrees per second).
            let white = g.add(WhiteNoise, [240.0, -180.0]);
            let angles = g.add(VectorMath, [480.0, -200.0]);
            let time = g.add(Time, [20.0, -60.0]);
            let spin = g.add(VectorMath, [700.0, -200.0]);
            let rot = g.add(RotateVector, [920.0, -320.0]);
            g.node_mut(angles).unwrap().set(0, &[3.0]); // scale
            g.node_mut(angles).unwrap().set(3, &[360.0]);
            g.node_mut(time).unwrap().set(0, &[12.0]);
            g.connect(rep, 1, white, 0);
            g.connect(white, 1, angles, 1);
            g.connect(angles, 0, spin, 1);
            g.connect(time, 0, spin, 2);
            g.connect(rep, 0, rot, 0);
            g.connect(spin, 0, rot, 1);
            g.connect(rot, 0, dout, 0);
            8
        }
        // ---- Landscapes: fractals made endless across the ground plane ----
        "Menger canyons" | "Molten canyons" | "Frozen catacombs" => {
            // A Menger sponge tiled across x and z: one endless slab cut by square canyons.
            single(&mut g, Menger);
            set_out(&mut g, 1024.0, 2.0, true);
            add_floor(&mut g, 2.0, false, 60.0);
            g.camera_hint = Some(match name {
                "Menger canyons" => ([0.17, 1.35, -4.0], 0.35, -0.3),
                "Molten canyons" => ([0.0, 0.62, -3.0], 0.0, -0.12), // in a canyon, above the lava
                _ => ([0.0, 0.62, -3.0], 0.0, 0.2), // in the tunnels, looking toward the snowy ledges
            });
            8
        }
        "Menger corridor" => {
            // Tiled along z only: the sponge's central square hole runs on forever.
            single(&mut g, Menger);
            set_out(&mut g, 1024.0, 2.0, true);
            let (rep, _) = add_repeat(&mut g, 2.0, false);
            g.node_mut(rep).unwrap().set(1, &[FLOOR_CELL, FLOOR_CELL, 2.0]);
            g.camera_hint = Some(([0.0, -0.12, 0.0], 0.0, 0.0));
            8
        }
        "Mandelbox highlands" => {
            // Positive-scale Mandelbox: a cube whose top face is a fractal city. Tiled
            // with mirroring, the faces meet seamlessly into endless snowy uplands.
            let b = single(&mut g, Mandelbox);
            g.node_mut(b).unwrap().set(1, &[2.0]);
            set_out(&mut g, 1024.0, 7.0, true);
            add_floor(&mut g, 12.0, true, 150.0);
            g.camera_hint = Some(([1.3, 6.6, -14.0], 0.5, -0.1));
            14
        }
        "Sierpinski lattice" => {
            single(&mut g, Sierpinski);
            set_out(&mut g, 1024.0, 2.0, true);
            add_floor(&mut g, 2.0, true, 60.0);
            g.camera_hint = Some(([0.3, 1.7, -4.0], 0.6, -0.35));
            14
        }
        "Bulb garden" => {
            // One layer of bulbs across the plane, each cell turned its own way
            // (Cell -> White Noise -> angles) and bobbing to its own height.
            single(&mut g, Mandelbulb);
            set_out(&mut g, 4.0, 1.5, false);
            let (rep, dout) = add_floor(&mut g, 2.6, false, 60.0);
            let white = g.add(WhiteNoise, [240.0, -180.0]);
            let angles = g.add(VectorMath, [480.0, -200.0]);
            let rot = g.add(RotateVector, [920.0, -320.0]);
            g.node_mut(angles).unwrap().set(0, &[3.0]); // scale
            g.node_mut(angles).unwrap().set(3, &[360.0]);
            g.connect(rep, 1, white, 0);
            g.connect(white, 1, angles, 1);
            g.connect(rep, 0, rot, 0);
            g.connect(angles, 0, rot, 1);
            g.connect(rot, 0, dout, 0);
            g.camera_hint = Some(([0.0, 2.4, -6.0], 0.45, -0.38));
            10
        }
        "Icosahedral glacier" => {
            let n = single(&mut g, PolyIfs);
            let p = g.node_mut(n).unwrap();
            p.set(2, &[2.2]);
            p.set(3, &[1.0, 0.4, 0.0]);
            p.set(4, &[12.0]);
            set_out(&mut g, 1024.0, 1.5, true);
            add_floor(&mut g, 1.6, true, 60.0);
            g.camera_hint = Some(([0.3, 1.5, -3.5], 0.5, -0.3));
            9
        }
        "Kleinian badlands" => {
            let k = single(&mut g, PseudoKleinian);
            g.node_mut(k).unwrap().set(1, &[0.92436, FLOOR_CELL, 0.92436]);
            set_out_mode(&mut g, 4096.0, 2.0, DE_KLEINIAN);
            add_view_distance(&mut g, 40.0);
            g.camera_hint = Some(([0.3, 1.5, -2.0], 0.5, -0.3));
            12
        }
        "Pagoda fields" => {
            single(&mut g, Kifs);
            set_out(&mut g, 1024.0, 1.5, true);
            add_floor(&mut g, 2.0, true, 60.0);
            g.camera_hint = Some(([0.3, 1.8, -4.0], 0.5, -0.35));
            12
        }
        "Mandelbox peaks" => {
            // Mandelbox cubes stood on a corner, one per cell: a range of fractal peaks.
            // Half-size 2 makes the half-diagonal 3.46, so cells of 7.2 keep them apart.
            let b = single(&mut g, Mandelbox);
            g.node_mut(b).unwrap().set(1, &[-1.5]);
            set_out(&mut g, 1024.0, 4.5, true);
            let (rep, dout) = add_floor(&mut g, 7.2, false, 120.0);
            let rot = g.add(RotateVector, [500.0, -260.0]);
            g.node_mut(rot).unwrap().set(1, &[35.264, 0.0, 45.0]);
            g.connect(rep, 0, rot, 0);
            g.connect(rot, 0, dout, 0);
            g.camera_hint = Some(([1.0, 4.5, -12.0], 0.3, -0.18));
            16
        }
        "Kleinian lagoon" => {
            // Wider boxes than the badlands: the sinkholes open into broad round pools.
            let k = single(&mut g, PseudoKleinian);
            g.node_mut(k).unwrap().set(1, &[1.0, FLOOR_CELL, 1.0]);
            g.node_mut(k).unwrap().set(2, &[1.15]);
            set_out_mode(&mut g, 4096.0, 2.0, DE_KLEINIAN);
            add_view_distance(&mut g, 40.0);
            g.camera_hint = Some(([0.3, 0.9, -2.0], 0.5, -0.25));
            12
        }
        _ => {
            single(&mut g, Mandelbulb);
            set_out(&mut g, 4.0, 1.5, false);
            10
        }
    };
    let (orbit, pal, material) = add_default_coloring(&mut g);
    match name {
        "Kleinian dreams" => {
            rainbow(&mut g, pal, 0.03);
            glow(&mut g, orbit, 3, material, 0.02, 4.0, [0.5, 0.8, 1.0]);
        }
        "Apollonian cathedral" => {
            glow(&mut g, orbit, 1, material, 0.015, 6.0, [1.0, 0.55, 0.2]);
        }
        "Morphing Mandelbox" => {
            rainbow(&mut g, pal, 0.02);
            let out = g.node_mut(material).unwrap();
            out.set(1, &[0.4]);
            out.set(2, &[0.3]);
        }
        "Psychedelic Juliabulb" => {
            rainbow(&mut g, pal, 0.05);
            glow(&mut g, orbit, 1, material, 0.01, 5.0, [1.0, 0.3, 0.9]);
        }
        "Infinite fly-through" => {
            rainbow(&mut g, pal, 0.04);
            glow(&mut g, orbit, 1, material, 0.004, 3.0, [0.3, 0.9, 1.0]);
        }
        "Kaleidoscope" => {
            rainbow(&mut g, pal, 0.06);
            glow(&mut g, orbit, 2, material, 0.02, 5.0, [1.0, 0.85, 0.4]);
        }
        "Snowflake Mandelbox" => {
            rainbow(&mut g, pal, 0.03);
            // Trap Z: trap Y would light up the mirror seams, where the fold puts y = 0.
            glow(&mut g, orbit, 3, material, 0.02, 4.0, [0.6, 0.85, 1.0]);
        }
        "Xenodream drift" => {
            rainbow(&mut g, pal, 0.04);
            glow(&mut g, orbit, 1, material, 0.01, 5.0, [1.0, 0.4, 0.8]);
        }
        "Tumbling bulb field" => {
            rainbow(&mut g, pal, 0.02);
        }
        "Lava bulb" => lava(&mut g, material, 9.0, 8.0),
        "Molten canyons" => lava(&mut g, material, 20.0, 2.5),
        "Frozen catacombs" => {
            // Dark basalt walls, snow on every ledge.
            let p = g.node_mut(pal).unwrap();
            p.set(1, &[0.16, 0.16, 0.18]);
            p.set(2, &[0.06, 0.06, 0.08]);
            snow_caps(&mut g, material, 0.6, 0.9);
        }
        "Menger canyons" => {
            // Sandstone: warm, low-contrast bands by orbit trap.
            let p = g.node_mut(pal).unwrap();
            p.set(1, &[0.42, 0.26, 0.15]);
            p.set(2, &[0.16, 0.1, 0.06]);
            p.set(3, &[1.4, 1.4, 1.4]);
            p.set(4, &[0.0, 0.05, 0.1]);
            g.node_mut(material).unwrap().set(2, &[0.75]);
        }
        "Menger corridor" => {
            rainbow(&mut g, pal, 0.02);
            glow(&mut g, orbit, 1, material, 0.008, 4.0, [0.3, 0.95, 1.0]);
        }
        "Mandelbox peaks" => snow_caps(&mut g, material, 0.35, 0.6),
        "Bulb garden" => rainbow(&mut g, pal, 0.01),
        "Kleinian badlands" => {
            // Salt flats: pale crust, the rims of the sinkholes lit cyan.
            let p = g.node_mut(pal).unwrap();
            p.set(1, &[0.7, 0.68, 0.64]);
            p.set(2, &[0.2, 0.2, 0.22]);
            glow(&mut g, orbit, 3, material, 0.01, 3.0, [0.35, 0.85, 1.0]);
        }
        "Kleinian lagoon" => {
            // Polished dark-teal stone that mirrors the sky, rims glowing gold.
            g.disconnect(material, 0);
            let out = g.node_mut(material).unwrap();
            out.set(0, &[0.04, 0.12, 0.14]);
            out.set(1, &[0.3]);
            out.set(2, &[0.08]);
            glow(&mut g, orbit, 3, material, 0.008, 4.0, [1.0, 0.75, 0.35]);
        }
        "Icosahedral glacier" => {
            // Ice: pale blue with light scattering through it.
            g.disconnect(material, 0);
            let out = g.node_mut(material).unwrap();
            out.set(0, &[0.62, 0.78, 0.9]);
            out.set(2, &[0.2]);
            out.set(6, &[0.8]);
            out.set(7, &[0.4, 0.75, 1.0]);
            out.set(8, &[0.8]);
        }
        "Beat-synced bulb" => {
            // Glowing veins whose brightness follows the bass.
            rainbow(&mut g, pal, 0.03);
            glow(&mut g, orbit, 1, material, 0.012, 12.0, [1.0, 0.35, 0.15]);
            let veins = g.source(material, 5).unwrap().0;
            let bass = g.add(AudioLevel, [20.0, 900.0]);
            let b = g.node_mut(bass).unwrap();
            b.set(0, &[1.0]);
            b.set(1, &[0.01]);
            b.set(2, &[0.3]);
            let mul = g.add(Math, [560.0, 760.0]);
            g.node_mut(mul).unwrap().set(0, &[2.0]);
            g.connect(veins, 0, mul, 1);
            g.connect(bass, 0, mul, 2);
            g.connect(mul, 0, material, 5);
        }
        "Glowing bulb" => {
            // Orbit trap X -> Map Range (small trap = hot) -> emission strength: glowing veins.
            let map = g.add(MapRange, [280.0, 640.0]);
            let m = g.node_mut(map).unwrap();
            m.set(1, &[0.0]);
            m.set(2, &[0.012]);
            m.set(3, &[10.0]);
            m.set(4, &[0.0]);
            g.connect(orbit, 1, map, 0);
            g.connect(map, 0, material, 5);
            let out = g.node_mut(material).unwrap();
            out.set(4, &[1.0, 0.45, 0.12]);
            out.set(2, &[0.6]);
        }
        "Chrome Mandelbox" => {
            g.disconnect(material, 0);
            let out = g.node_mut(material).unwrap();
            out.set(0, &[0.92, 0.92, 0.95]);
            out.set(1, &[1.0]);
            out.set(2, &[0.12]);
        }
        "Jade bulb" => {
            g.disconnect(material, 0);
            let out = g.node_mut(material).unwrap();
            out.set(0, &[0.12, 0.42, 0.25]);
            out.set(2, &[0.25]);
            out.set(6, &[0.85]);
            out.set(7, &[0.35, 0.95, 0.55]);
            out.set(8, &[0.7]);
        }
        _ => {}
    }
    (g, iters)
}

/// Position -> Repeat -> Domain Output, laid out above the iteration graph.
/// Returns (repeat, domain output).
fn add_repeat(g: &mut Graph, cell: f32, mirror: bool) -> (NodeId, NodeId) {
    use NodeKind::*;
    let p = g.add(Position, [20.0, -320.0]);
    let rep = g.add(Repeat, [240.0, -340.0]);
    let out = g.add(DomainOutput, [760.0, -320.0]);
    let r = g.node_mut(rep).unwrap();
    r.set(1, &[cell; 3]);
    r.set(3, &[mirror as u32 as f32]);
    g.connect(p, 0, rep, 0);
    g.connect(rep, 0, out, 0);
    (rep, out)
}

/// Cell size for an axis that shouldn't repeat: far beyond anything the camera reaches.
/// (Repeat's limit can't say "no copies" exactly: WGSL and Rust round 0.5 differently.)
const FLOOR_CELL: f32 = 1.0e4;

/// Repeat across x and z only, so the fractal becomes an endless floor.
/// Returns (repeat, domain output).
fn add_floor(g: &mut Graph, cell: f32, mirror: bool, view_distance: f32) -> (NodeId, NodeId) {
    let (rep, out) = add_repeat(g, cell, mirror);
    g.node_mut(rep).unwrap().set(1, &[cell, FLOOR_CELL, cell]);
    g.node_mut(out).unwrap().set(1, &[view_distance]);
    (rep, out)
}

/// A plain Domain Output, just to set how far an unbounded scene is marched.
fn add_view_distance(g: &mut Graph, view_distance: f32) {
    let p = g.add(NodeKind::Position, [20.0, -320.0]);
    let out = g.add(NodeKind::DomainOutput, [760.0, -320.0]);
    g.connect(p, 0, out, 0);
    g.node_mut(out).unwrap().set(1, &[view_distance]);
}

/// Oscillator node: sine at `freq` Hz, `center` +- `amp`.
fn oscillator(g: &mut Graph, pos: [f32; 2], freq: f32, amp: f32, center: f32) -> NodeId {
    let o = g.add(NodeKind::Oscillator, pos);
    let n = g.node_mut(o).unwrap();
    n.set(1, &[freq]);
    n.set(2, &[amp]);
    n.set(3, &[center]);
    o
}

/// Turns the palette into iq's rainbow and cycles its hue: Time -> palette phase.
fn rainbow(g: &mut Graph, pal: NodeId, cycles_per_second: f32) {
    let p = g.node_mut(pal).unwrap();
    p.set(1, &[0.5, 0.5, 0.5]);
    p.set(2, &[0.5, 0.5, 0.5]);
    p.set(3, &[1.0, 1.0, 1.0]);
    p.set(4, &[0.0, 0.33, 0.67]);
    let time = g.add(NodeKind::Time, [20.0, 560.0]);
    g.node_mut(time).unwrap().set(0, &[cycles_per_second]);
    let add = g.add(NodeKind::VectorMath, [280.0, 640.0]);
    g.node_mut(add).unwrap().set(2, &[0.0, 0.33, 0.67]);
    g.connect(time, 0, add, 1);
    g.connect(add, 0, pal, 4);
}

/// Glowing veins: where orbit trap `trap` (an Orbit Data output) is below `width`,
/// emission rises to `strength` in `color`.
fn glow(g: &mut Graph, orbit: NodeId, trap: usize, material: NodeId, width: f32, strength: f32, color: [f32; 3]) {
    let map = g.add(NodeKind::MapRange, [280.0, 760.0]);
    let m = g.node_mut(map).unwrap();
    m.set(1, &[0.0]);
    m.set(2, &[width]);
    m.set(3, &[strength]);
    m.set(4, &[0.0]);
    g.connect(orbit, trap, map, 0);
    g.connect(map, 0, material, 5);
    g.node_mut(material).unwrap().set(4, &color);
}

/// Dark crust split by Voronoi cracks (`scale` cells per unit) glowing up to `strength`, at a blackbody
/// temperature drifting with noise. Replaces the palette coloring, so no Orbit Data is
/// upstream and the orbit pass compiles out.
fn lava(g: &mut Graph, material: NodeId, scale: f32, strength: f32) {
    use NodeKind::*;
    g.disconnect(material, 0);
    let vor = g.add(VoronoiTexture, [20.0, 900.0]);
    g.node_mut(vor).unwrap().set(1, &[scale]);
    let cracks = g.add(MapRange, [280.0, 900.0]);
    let m = g.node_mut(cracks).unwrap();
    m.set(2, &[0.08]); // edge distance 0..0.08 -> strength..0
    m.set(3, &[strength]);
    m.set(4, &[0.0]);
    g.connect(vor, 1, cracks, 0);
    g.connect(cracks, 0, material, 5);
    let noise = g.add(NoiseTexture, [20.0, 1120.0]);
    g.node_mut(noise).unwrap().set(1, &[3.0]);
    let temp = g.add(MapRange, [280.0, 1120.0]);
    g.node_mut(temp).unwrap().set(3, &[900.0]); // fac 0..1 -> 900..3200 K
    g.node_mut(temp).unwrap().set(4, &[3200.0]);
    let bb = g.add(Blackbody, [520.0, 1120.0]);
    g.connect(noise, 0, temp, 0);
    g.connect(temp, 0, bb, 0);
    g.connect(bb, 0, material, 4);
    let out = g.node_mut(material).unwrap();
    out.set(0, &[0.035, 0.03, 0.03]);
    out.set(2, &[0.85]);
}

/// Snow on whatever faces up: Geometry normal Y (from `lo` bare to `hi` fully white)
/// -> Map Range -> mix of the current base color (rock) and white.
fn snow_caps(g: &mut Graph, material: NodeId, lo: f32, hi: f32) {
    use NodeKind::*;
    let (rock, rock_out) = g.source(material, 0).unwrap();
    let geo = g.add(Geometry, [20.0, 1000.0]);
    let sep = g.add(SeparateXyz, [240.0, 1000.0]);
    let up = g.add(MapRange, [460.0, 1000.0]);
    let mix = g.add(MixColor, [700.0, 640.0]);
    let m = g.node_mut(up).unwrap();
    m.set(1, &[lo]);
    m.set(2, &[hi]);
    g.node_mut(mix).unwrap().set(2, &[0.93, 0.95, 1.0]);
    g.connect(geo, 1, sep, 0);
    g.connect(sep, 1, up, 0);
    g.connect(up, 0, mix, 0);
    g.connect(rock, rock_out, mix, 1);
    g.connect(mix, 0, material, 0);
}

/// Orbit trap radius -> cosine palette -> Material Output, laid out below the iteration
/// graph. Returns (orbit data, palette, material output).
fn add_default_coloring(g: &mut Graph) -> (NodeId, NodeId, NodeId) {
    use NodeKind::*;
    let orbit = g.add(OrbitData, [20.0, 420.0]);
    let pal = g.add(CosinePalette, [280.0, 400.0]);
    let out = g.add(ColorOutput, [560.0, 420.0]);
    let p = g.node_mut(pal).unwrap();
    p.set(1, &[0.45, 0.4, 0.4]);
    p.set(2, &[0.4, 0.3, 0.3]);
    p.set(3, &[1.0, 1.0, 1.0]);
    p.set(4, &[0.0, 0.15, 0.3]);
    g.connect(orbit, 0, pal, 0);
    g.connect(pal, 0, out, 0);
    (orbit, pal, out)
}
