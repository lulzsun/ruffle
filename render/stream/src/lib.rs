//! A Ruffle render backend that emits a draw-command stream instead of
//! rasterizing.
//!
//! Ruffle's `RenderBackend` trait already hands a backend everything a
//! renderer needs and nothing it doesn't: `register_shape` delivers vector
//! geometry (which we tessellate once, with Ruffle's own tessellator), and
//! `submit_frame` delivers an already-culled `CommandList` describing what to
//! draw this frame. Neither requires that we own a GPU context.
//!
//! So this backend owns no context. It flattens both into the plain data types
//! in [`sink`] and hands them to a [`CommandSink`]. The host — three.js, in
//! whirled2's case — draws them in a context it already has, which removes the
//! per-avatar canvas, the GPU readback, and the texture upload that the
//! screen-scraping approach needs.
//!
//! See `docs/specs/swf-avatar-rendering.md` (W2) in the whirled2 repo.

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

use ruffle_render::backend::{
    BitmapCacheEntry, Context3D, Context3DProfile, PixelBenderOutput, PixelBenderTarget,
    RenderBackend, ShapeHandle, ShapeHandleImpl, ViewportDimensions,
};
use ruffle_render::bitmap::{
    Bitmap, BitmapHandle, BitmapHandleImpl, BitmapSource, PixelRegion, RgbaBufRead, SyncHandle,
};
use ruffle_render::commands::{Command, CommandList, RenderBlendMode};
use ruffle_render::error::Error;
use ruffle_render::pixel_bender::{PixelBenderShader, PixelBenderShaderHandle};
use ruffle_render::pixel_bender_support::PixelBenderShaderArgument;
use ruffle_render::quality::StageQuality;
use ruffle_render::shape_utils::{DistilledShape, GradientType};
use ruffle_render::tessellator::{DrawType, ShapeTessellator};
use ruffle_render::transform::Transform;

pub mod sink;

#[cfg(feature = "web")]
pub mod js_sink;

#[cfg(feature = "web")]
pub use js_sink::JsCommandSink;

pub use sink::{
    CommandSink, DrawKind, DrawTransform, FrameCommand, GradientSpec, LoggingSink, RecordingSink,
    ShapeDraw,
};

/// Identifies a shape registered with this backend.
pub type ShapeId = u32;

/// Identifies a bitmap registered with this backend.
pub type BitmapId = u32;

#[derive(Debug)]
struct StreamShape {
    id: ShapeId,
}

impl ShapeHandleImpl for StreamShape {}

fn as_stream_shape(handle: &ShapeHandle) -> &StreamShape {
    <dyn Any>::downcast_ref(&*handle.0).expect("Shape handle must be a stream StreamShape")
}

#[derive(Debug)]
struct StreamBitmap {
    id: BitmapId,
    #[expect(dead_code, reason = "kept for diagnostics and future frame culling")]
    width: u32,
    #[expect(dead_code, reason = "kept for diagnostics and future frame culling")]
    height: u32,
}

impl BitmapHandleImpl for StreamBitmap {}

fn as_stream_bitmap(handle: &BitmapHandle) -> &StreamBitmap {
    <dyn Any>::downcast_ref(&*handle.0).expect("Bitmap handle must be a stream StreamBitmap")
}

pub struct StreamRenderBackend {
    viewport: ViewportDimensions,
    tessellator: ShapeTessellator,
    sink: Box<dyn CommandSink>,
    next_shape_id: ShapeId,
    next_bitmap_id: BitmapId,
}

impl StreamRenderBackend {
    pub fn new(sink: Box<dyn CommandSink>) -> Self {
        Self {
            viewport: ViewportDimensions {
                width: 0,
                height: 0,
                scale_factor: 1.0,
            },
            tessellator: ShapeTessellator::new(),
            sink,
            next_shape_id: 0,
            next_bitmap_id: 0,
        }
    }

    /// Convenience constructor for the M2 stub, which only logs.
    pub fn logging() -> Self {
        Self::new(Box::new(LoggingSink::new()))
    }

    /// Flatten a Ruffle `Transform` into the sink's plain representation.
    fn flatten_transform(transform: &Transform) -> DrawTransform {
        let m = &transform.matrix;
        let c = &transform.color_transform;
        DrawTransform {
            matrix: [
                m.a,
                m.b,
                m.c,
                m.d,
                m.tx.get() as f32,
                m.ty.get() as f32,
            ],
            mult: c.mult_rgba_normalized(),
            add: c.add_rgba_normalized(),
        }
    }

    /// Walk a `CommandList` into a flat draw list.
    ///
    /// Nested lists (masks and blends) are inlined between marker commands
    /// rather than kept as a tree, so the consumer can execute the stream with
    /// a simple loop and a small state stack.
    fn flatten_commands(&self, list: &CommandList, out: &mut Vec<FrameCommand>) {
        for command in &list.commands {
            match command {
                Command::RenderShape { shape, transform } => {
                    out.push(FrameCommand::DrawShape {
                        shape: as_stream_shape(shape).id,
                        transform: Self::flatten_transform(transform),
                    });
                }
                Command::RenderBitmap {
                    bitmap,
                    transform,
                    smoothing,
                    ..
                } => {
                    out.push(FrameCommand::DrawBitmap {
                        bitmap: as_stream_bitmap(bitmap).id,
                        transform: Self::flatten_transform(transform),
                        smoothing: *smoothing,
                    });
                }
                Command::DrawRect { color, matrix } => {
                    out.push(FrameCommand::DrawRect {
                        color: [
                            f32::from(color.r) / 255.0,
                            f32::from(color.g) / 255.0,
                            f32::from(color.b) / 255.0,
                            f32::from(color.a) / 255.0,
                        ],
                        matrix: [
                            matrix.a,
                            matrix.b,
                            matrix.c,
                            matrix.d,
                            matrix.tx.get() as f32,
                            matrix.ty.get() as f32,
                        ],
                    });
                }
                Command::RenderAlphaMask {
                    maskee_commands,
                    mask_commands,
                } => {
                    out.push(FrameCommand::PushMask);
                    self.flatten_commands(mask_commands, out);
                    out.push(FrameCommand::ActivateMask);
                    self.flatten_commands(maskee_commands, out);
                    out.push(FrameCommand::DeactivateMask);
                    out.push(FrameCommand::PopMask);
                }
                Command::PushMask => out.push(FrameCommand::PushMask),
                Command::ActivateMask => out.push(FrameCommand::ActivateMask),
                Command::DeactivateMask => out.push(FrameCommand::DeactivateMask),
                Command::PopMask => out.push(FrameCommand::PopMask),
                Command::Blend(commands, mode) => {
                    let mode = match mode {
                        RenderBlendMode::Builtin(mode) => *mode,
                        // Shader blends need PixelBender, which this backend
                        // declines to support. Fall back to normal compositing
                        // rather than dropping the content entirely.
                        RenderBlendMode::Shader(_) => swf::BlendMode::Normal,
                    };
                    out.push(FrameCommand::BeginBlend { mode });
                    self.flatten_commands(commands, out);
                    out.push(FrameCommand::EndBlend);
                }
                // Stage3D is a non-goal; there is nothing sensible to emit.
                Command::RenderStage3D { .. } => {}
                // Line drawing is only used by Ruffle's own debug overlays.
                Command::DrawLine { .. } | Command::DrawLineRect { .. } => {}
            }
        }
    }
}

impl RenderBackend for StreamRenderBackend {
    fn viewport_dimensions(&self) -> ViewportDimensions {
        self.viewport
    }

    fn set_viewport_dimensions(&mut self, dimensions: ViewportDimensions) {
        self.viewport = dimensions;
    }

    fn register_shape(
        &mut self,
        shape: DistilledShape,
        bitmap_source: &dyn BitmapSource,
    ) -> ShapeHandle {
        let id = self.next_shape_id;
        self.next_shape_id += 1;

        let mesh = self.tessellator.tessellate_shape(shape, bitmap_source);

        let mut draws = Vec::with_capacity(mesh.draws.len());
        for draw in mesh.draws {
            // Interleave into [x, y, r, g, b, a] with colors normalized,
            // which is what a GPU vertex buffer wants anyway.
            let mut vertices = Vec::with_capacity(draw.vertices.len() * 6);
            for v in &draw.vertices {
                vertices.extend_from_slice(&[
                    v.x,
                    v.y,
                    f32::from(v.color.r) / 255.0,
                    f32::from(v.color.g) / 255.0,
                    f32::from(v.color.b) / 255.0,
                    f32::from(v.color.a) / 255.0,
                ]);
            }

            let kind = match draw.draw_type {
                DrawType::Color => DrawKind::Color,
                DrawType::Gradient { matrix, gradient } => DrawKind::Gradient { matrix, gradient },
                DrawType::Bitmap(bitmap) => {
                    // A bitmap fill names a SWF character id, not a handle.
                    // Resolving it can register the bitmap with us as a side
                    // effect, which is why this needs `&mut self`.
                    let resolved = bitmap_source.bitmap_handle(bitmap.bitmap_id, self);
                    match resolved {
                        Some(handle) => DrawKind::Bitmap {
                            bitmap: as_stream_bitmap(&handle).id,
                            matrix: bitmap.matrix,
                            is_smoothed: bitmap.is_smoothed,
                            is_repeating: bitmap.is_repeating,
                        },
                        None => {
                            // Flash tolerates a fill that names a missing
                            // bitmap; draw it untextured rather than dropping
                            // the geometry.
                            tracing::warn!(
                                "stream: shape {} references unknown bitmap id {}",
                                id,
                                bitmap.bitmap_id
                            );
                            DrawKind::Color
                        }
                    }
                }
            };

            draws.push(ShapeDraw {
                kind,
                vertices,
                indices: draw.indices,
            });
        }

        // Gradient fills reference these by index. Encodings follow Ruffle's
        // own gradient shader; see GradientSpec.
        let gradients = mesh
            .gradients
            .into_iter()
            .map(|gradient| GradientSpec {
                kind: match gradient.gradient_type {
                    GradientType::Linear => 0,
                    GradientType::Radial => 1,
                    GradientType::Focal => 2,
                },
                repeat: match gradient.repeat_mode {
                    swf::GradientSpread::Pad => 0,
                    swf::GradientSpread::Repeat => 1,
                    swf::GradientSpread::Reflect => 2,
                },
                interpolation: match gradient.interpolation {
                    swf::GradientInterpolation::Rgb => 0,
                    swf::GradientInterpolation::LinearRgb => 1,
                },
                // Matches the clamp Ruffle applies; the focal maths degenerates
                // at exactly +/-1.
                focal_point: gradient.focal_point.to_f32().clamp(-0.98, 0.98),
                stops: gradient
                    .records
                    .iter()
                    .map(|record| {
                        (
                            f32::from(record.ratio) / 255.0,
                            [
                                f32::from(record.color.r) / 255.0,
                                f32::from(record.color.g) / 255.0,
                                f32::from(record.color.b) / 255.0,
                                f32::from(record.color.a) / 255.0,
                            ],
                        )
                    })
                    .collect(),
            })
            .collect();

        self.sink.register_shape(id, draws, gradients);
        ShapeHandle(Arc::new(StreamShape { id }))
    }

    fn submit_frame(
        &mut self,
        clear: swf::Color,
        commands: CommandList,
        _cache_entries: Vec<BitmapCacheEntry>,
    ) {
        let mut flat = Vec::with_capacity(commands.commands.len());
        self.flatten_commands(&commands, &mut flat);

        self.sink.submit_frame(
            [
                f32::from(clear.r) / 255.0,
                f32::from(clear.g) / 255.0,
                f32::from(clear.b) / 255.0,
                f32::from(clear.a) / 255.0,
            ],
            (self.viewport.width, self.viewport.height),
            flat,
        );
    }

    fn register_bitmap(&mut self, bitmap: Bitmap<'_>) -> Result<BitmapHandle, Error> {
        let id = self.next_bitmap_id;
        self.next_bitmap_id += 1;

        let width = bitmap.width();
        let height = bitmap.height();
        let rgba = bitmap.to_rgba().data().to_vec();

        self.sink.register_bitmap(id, width, height, rgba);
        Ok(BitmapHandle(Arc::new(StreamBitmap { id, width, height })))
    }

    fn update_texture(
        &mut self,
        handle: &BitmapHandle,
        bitmap: Bitmap<'_>,
        _region: PixelRegion,
    ) -> Result<(), Error> {
        let id = as_stream_bitmap(handle).id;
        let width = bitmap.width();
        let height = bitmap.height();
        let rgba = bitmap.to_rgba().data().to_vec();

        // Always send the whole bitmap for now. Honouring `region` is a
        // bandwidth optimization for M3, not a correctness issue.
        self.sink.update_bitmap(id, width, height, rgba);
        Ok(())
    }

    fn create_empty_texture(&mut self, width: u32, height: u32) -> Result<BitmapHandle, Error> {
        let id = self.next_bitmap_id;
        self.next_bitmap_id += 1;

        let rgba = vec![0; (width as usize) * (height as usize) * 4];
        self.sink.register_bitmap(id, width, height, rgba);
        Ok(BitmapHandle(Arc::new(StreamBitmap { id, width, height })))
    }

    // --- Deliberately unsupported ------------------------------------------
    //
    // Returning None/false here is the supported way to declare a limited
    // backend: Ruffle degrades rather than panicking. Filters are the one
    // entry here we expect to revisit, since glow and drop-shadow are common
    // in real avatars.

    fn render_offscreen(
        &mut self,
        _handle: BitmapHandle,
        _commands: CommandList,
        _quality: StageQuality,
        _bounds: PixelRegion,
    ) -> Option<Box<dyn SyncHandle>> {
        None
    }

    fn create_context3d(
        &mut self,
        _profile: Context3DProfile,
    ) -> Result<Box<dyn Context3D>, Error> {
        Err(Error::Unimplemented("Stage3D".into()))
    }

    fn compile_pixelbender_shader(
        &mut self,
        _shader: PixelBenderShader,
    ) -> Result<PixelBenderShaderHandle, Error> {
        Err(Error::Unimplemented("PixelBender".into()))
    }

    fn run_pixelbender_shader(
        &mut self,
        _handle: PixelBenderShaderHandle,
        _arguments: &[PixelBenderShaderArgument],
        _target: &PixelBenderTarget,
    ) -> Result<PixelBenderOutput, Error> {
        Err(Error::Unimplemented("PixelBender".into()))
    }

    fn resolve_sync_handle(
        &mut self,
        _handle: Box<dyn SyncHandle>,
        _with_rgba: RgbaBufRead,
    ) -> Result<(), Error> {
        Err(Error::Unimplemented("Offscreen rendering".into()))
    }

    fn debug_info(&self) -> Cow<'static, str> {
        Cow::Borrowed("Command stream (no rasterization)")
    }

    fn name(&self) -> &'static str {
        "stream"
    }

    fn set_quality(&mut self, _quality: StageQuality) {
        // Quality affects antialiasing during rasterization, which we do not
        // do. The consumer decides its own sampling.
    }
}
