//! The seam between the backend and whatever consumes the command stream.
//!
//! Keeping this a trait means the backend can be exercised from a plain
//! `cargo test` with a recording sink, without a browser or a JS host in the
//! picture. The wasm host installs its own sink that forwards to JS.

use crate::{BitmapId, ShapeId};

/// A gradient fill definition, referenced by index from `DrawKind::Gradient`.
///
/// Field encodings match Ruffle's own gradient shader so the consumer can
/// reproduce it exactly rather than approximating.
#[derive(Debug, Clone)]
pub struct GradientSpec {
    /// 0 linear, 1 radial, 2 focal.
    pub kind: u8,
    /// 0 clamp (Pad), 1 repeat (Repeat), 2 mirror (Reflect).
    ///
    /// Note this is not the same ordering as `swf::GradientSpread`, which is
    /// Pad/Reflect/Repeat — the remap matches what Ruffle's shader expects.
    pub repeat: u8,
    /// 0 sRGB, 1 linear RGB. When 1, stops must be converted to linear space
    /// before interpolating and the result converted back afterwards.
    pub interpolation: u8,
    /// Focal point for `kind == 2`, clamped to Ruffle's usable range.
    pub focal_point: f32,
    /// `(ratio 0..1, rgba 0..1)` stops, in order.
    pub stops: Vec<(f32, [f32; 4])>,
}

/// One tessellated sub-draw of a registered shape.
///
/// Vertices are interleaved `[x, y, r, g, b, a]` with colors already
/// normalized to 0..1, so the consumer can upload them to a vertex buffer
/// without touching them.
#[derive(Debug, Clone)]
pub struct ShapeDraw {
    pub kind: DrawKind,
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Which shader a sub-draw needs. Mirrors `tessellator::DrawType`, flattened
/// so the consumer does not need Ruffle's types.
#[derive(Debug, Clone)]
pub enum DrawKind {
    /// Per-vertex color only.
    Color,
    /// Sampled from a gradient ramp, with a texture-space matrix.
    Gradient { matrix: [[f32; 3]; 3], gradient: usize },
    /// Sampled from a registered bitmap, with a texture-space matrix.
    Bitmap {
        bitmap: BitmapId,
        matrix: [[f32; 3]; 3],
        is_smoothed: bool,
        is_repeating: bool,
    },
}

/// A 2D affine transform plus color transform, as Flash applies it.
///
/// `matrix` is `[a, b, c, d, tx, ty]`; `tx`/`ty` are in twips, matching
/// Ruffle's internal units. `mult` and `add` are RGBA color transform terms.
#[derive(Debug, Clone, Copy)]
pub struct DrawTransform {
    pub matrix: [f32; 6],
    pub mult: [f32; 4],
    pub add: [f32; 4],
}

/// One entry in a frame's draw list.
#[derive(Debug, Clone)]
pub enum FrameCommand {
    DrawShape { shape: ShapeId, transform: DrawTransform },
    DrawBitmap { bitmap: BitmapId, transform: DrawTransform, smoothing: bool },
    DrawRect { color: [f32; 4], matrix: [f32; 6] },
    /// Begin capturing draws that define a mask region.
    PushMask,
    /// Stop capturing; subsequent draws are masked by what was captured.
    ActivateMask,
    /// Stop applying the current mask.
    DeactivateMask,
    /// Discard the current mask.
    PopMask,
    /// Everything until the matching `EndBlend` composites with `mode`.
    ///
    /// Carries the typed mode rather than a name so consumers can map it to
    /// their own representation without string matching.
    BeginBlend { mode: swf::BlendMode },
    EndBlend,
}

/// Receives resource registrations and per-frame draw lists.
///
/// Resource calls are rare (once per shape or bitmap); `submit_frame` is called
/// once per SWF frame, not once per display frame.
pub trait CommandSink {
    /// A shape was tessellated. Sent once, before any frame references it.
    ///
    /// `gradients` is indexed by `DrawKind::Gradient::gradient`.
    fn register_shape(
        &mut self,
        id: ShapeId,
        draws: Vec<ShapeDraw>,
        gradients: Vec<GradientSpec>,
    );

    /// A bitmap's RGBA pixels. Sent once per bitmap, and again on update.
    fn register_bitmap(&mut self, id: BitmapId, width: u32, height: u32, rgba: Vec<u8>);

    /// A bitmap's pixels changed.
    fn update_bitmap(&mut self, id: BitmapId, width: u32, height: u32, rgba: Vec<u8>);

    /// One complete frame's draw list, in painter order.
    ///
    /// `viewport` is the pixel size Ruffle scaled its transforms into. The
    /// consumer must size its output and camera to match, or everything lands
    /// in the wrong place.
    fn submit_frame(
        &mut self,
        clear: [f32; 4],
        viewport: (u32, u32),
        commands: Vec<FrameCommand>,
    );
}

/// A sink that logs what it receives and keeps counters.
///
/// This is the M2 stub: it proves the command stream is being produced
/// correctly before any of it is wired to three.js.
#[derive(Debug, Default)]
pub struct LoggingSink {
    pub shapes_registered: usize,
    pub bitmaps_registered: usize,
    pub frames_submitted: usize,
    pub last_frame_commands: usize,
    /// Log every frame rather than only the first few. Very noisy.
    pub verbose: bool,
}

impl LoggingSink {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CommandSink for LoggingSink {
    fn register_shape(
        &mut self,
        id: ShapeId,
        draws: Vec<ShapeDraw>,
        gradients: Vec<GradientSpec>,
    ) {
        self.shapes_registered += 1;
        let _ = &gradients;
        let verts: usize = draws.iter().map(|d| d.vertices.len() / 6).sum();
        let indices: usize = draws.iter().map(|d| d.indices.len()).sum();
        tracing::info!(
            "stream: shape {} registered: {} draws, {} verts, {} indices, {} gradients",
            id,
            draws.len(),
            verts,
            indices,
            gradients.len()
        );
    }

    fn register_bitmap(&mut self, id: BitmapId, width: u32, height: u32, rgba: Vec<u8>) {
        self.bitmaps_registered += 1;
        tracing::info!(
            "stream: bitmap {} registered: {}x{} ({} bytes)",
            id,
            width,
            height,
            rgba.len()
        );
    }

    fn update_bitmap(&mut self, id: BitmapId, width: u32, height: u32, _rgba: Vec<u8>) {
        tracing::debug!("stream: bitmap {} updated: {}x{}", id, width, height);
    }

    fn submit_frame(
        &mut self,
        _clear: [f32; 4],
        _viewport: (u32, u32),
        commands: Vec<FrameCommand>,
    ) {
        self.frames_submitted += 1;
        self.last_frame_commands = commands.len();
        // The first few frames are the interesting ones; after that this would
        // drown the console at the SWF's frame rate.
        if self.verbose || self.frames_submitted <= 3 {
            tracing::info!(
                "stream: frame {} submitted: {} commands",
                self.frames_submitted,
                commands.len()
            );
        }
    }
}

/// A sink that records everything, for tests.
#[derive(Debug, Default)]
pub struct RecordingSink {
    pub shapes: Vec<(ShapeId, Vec<ShapeDraw>)>,
    pub bitmaps: Vec<(BitmapId, u32, u32)>,
    pub frames: Vec<Vec<FrameCommand>>,
}

impl CommandSink for RecordingSink {
    fn register_shape(
        &mut self,
        id: ShapeId,
        draws: Vec<ShapeDraw>,
        _gradients: Vec<GradientSpec>,
    ) {
        self.shapes.push((id, draws));
    }

    fn register_bitmap(&mut self, id: BitmapId, width: u32, height: u32, _rgba: Vec<u8>) {
        self.bitmaps.push((id, width, height));
    }

    fn update_bitmap(&mut self, _id: BitmapId, _width: u32, _height: u32, _rgba: Vec<u8>) {}

    fn submit_frame(
        &mut self,
        _clear: [f32; 4],
        _viewport: (u32, u32),
        commands: Vec<FrameCommand>,
    ) {
        self.frames.push(commands);
    }
}
