//! A [`CommandSink`] that hands the command stream to JavaScript.
//!
//! Bulk data (vertices, indices, pixels) crosses as typed arrays, and the
//! per-frame draw list crosses as one packed `Float32Array` of fixed-width
//! records rather than an array of objects — a frame with a few hundred draws
//! would otherwise allocate a few hundred JS objects at the SWF's frame rate.
//!
//! The host attaches a function named `whirledStream` to the Ruffle player
//! object before calling `load()`. If it is absent, events are dropped
//! silently: a page that has not opted in should not pay for this.

use js_sys::{Array, Float32Array, Function, Object, Reflect, Uint32Array, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};

use crate::sink::{CommandSink, DrawKind, FrameCommand, GradientSpec, ShapeDraw};
use crate::{BitmapId, ShapeId};

/// Event discriminants, mirrored by the TypeScript consumer.
const EVENT_SHAPE: f64 = 0.0;
const EVENT_BITMAP: f64 = 1.0;
const EVENT_FRAME: f64 = 2.0;
const EVENT_BITMAP_UPDATE: f64 = 3.0;

/// Draw-kind discriminants within a shape.
const KIND_COLOR: f64 = 0.0;
const KIND_GRADIENT: f64 = 1.0;
const KIND_BITMAP: f64 = 2.0;

/// Opcodes within a packed frame record.
const OP_DRAW_SHAPE: f32 = 0.0;
const OP_DRAW_BITMAP: f32 = 1.0;
const OP_DRAW_RECT: f32 = 2.0;
const OP_PUSH_MASK: f32 = 3.0;
const OP_ACTIVATE_MASK: f32 = 4.0;
const OP_DEACTIVATE_MASK: f32 = 5.0;
const OP_POP_MASK: f32 = 6.0;
const OP_BEGIN_BLEND: f32 = 7.0;
const OP_END_BLEND: f32 = 8.0;

/// Floats per packed frame record. Layout:
///
/// ```text
///  0      opcode
///  1      handle (shape id, bitmap id, or blend mode id)
///  2..8   matrix a, b, c, d, tx, ty
///  8..12  color transform multiply rgba
/// 12..16  color transform add rgba
/// 16      flags (bit 0: smoothing)
/// 17      reserved
/// ```
pub const RECORD_STRIDE: usize = 18;

pub struct JsCommandSink {
    /// The JS player object the host attached `whirledStream` to.
    target: JsValue,
}

impl JsCommandSink {
    pub fn new(target: JsValue) -> Self {
        Self { target }
    }

    /// Resolve the host callback, if the page installed one.
    ///
    /// Looked up per event rather than cached, so a host can attach or replace
    /// the handler at any point without the backend having to be rebuilt.
    ///
    /// `target` is Ruffle's internal InnerPlayer, which the embedding page has
    /// no reference to, so the handler is really expected on its `element` —
    /// the `<ruffle-player>` custom element the page created. Both are checked,
    /// so either works.
    fn callback(&self) -> Option<(Function, JsValue)> {
        if let Some(found) = lookup(&self.target) {
            return Some((found, self.target.clone()));
        }
        let element = Reflect::get(&self.target, &JsValue::from_str("element")).ok()?;
        lookup(&element).map(|found| (found, element))
    }

    fn emit(&self, event: &Object) {
        let Some((callback, this)) = self.callback() else {
            return;
        };
        if let Err(error) = callback.call1(&this, event) {
            tracing::warn!("stream: host whirledStream handler threw: {:?}", error);
        }
    }
}

/// Read a `whirledStream` function off a JS object, if it has one.
fn lookup(object: &JsValue) -> Option<Function> {
    if object.is_undefined() || object.is_null() {
        return None;
    }
    Reflect::get(object, &JsValue::from_str("whirledStream"))
        .ok()
        .and_then(|value| value.dyn_into::<Function>().ok())
}

fn set(object: &Object, key: &str, value: &JsValue) {
    // Only fails if `object` is not extensible, which it is by construction.
    let _ = Reflect::set(object, &JsValue::from_str(key), value);
}

fn set_f64(object: &Object, key: &str, value: f64) {
    set(object, key, &JsValue::from_f64(value));
}

/// Map a blend mode onto a stable integer the consumer can switch on.
fn blend_mode_id(mode: swf::BlendMode) -> f32 {
    use swf::BlendMode::*;
    match mode {
        Normal => 0.0,
        Layer => 1.0,
        Multiply => 2.0,
        Screen => 3.0,
        Lighten => 4.0,
        Darken => 5.0,
        Difference => 6.0,
        Add => 7.0,
        Subtract => 8.0,
        Invert => 9.0,
        Alpha => 10.0,
        Erase => 11.0,
        Overlay => 12.0,
        HardLight => 13.0,
    }
}

impl CommandSink for JsCommandSink {
    fn register_shape(
        &mut self,
        id: ShapeId,
        draws: Vec<ShapeDraw>,
        gradients: Vec<GradientSpec>,
    ) {
        if self.callback().is_none() {
            return;
        }

        let list = Array::new_with_length(draws.len() as u32);
        for (index, draw) in draws.iter().enumerate() {
            let entry = Object::new();

            match &draw.kind {
                DrawKind::Color => set_f64(&entry, "kind", KIND_COLOR),
                DrawKind::Gradient { matrix, gradient } => {
                    set_f64(&entry, "kind", KIND_GRADIENT);
                    set_f64(&entry, "gradient", *gradient as f64);
                    set(&entry, "matrix", &flatten_matrix3(matrix).into());
                }
                DrawKind::Bitmap {
                    bitmap,
                    matrix,
                    is_smoothed,
                    is_repeating,
                } => {
                    set_f64(&entry, "kind", KIND_BITMAP);
                    set_f64(&entry, "bitmap", *bitmap as f64);
                    set(&entry, "matrix", &flatten_matrix3(matrix).into());
                    set(&entry, "smoothed", &JsValue::from_bool(*is_smoothed));
                    set(&entry, "repeating", &JsValue::from_bool(*is_repeating));
                }
            }

            set(
                &entry,
                "vertices",
                &Float32Array::from(draw.vertices.as_slice()).into(),
            );
            set(
                &entry,
                "indices",
                &Uint32Array::from(draw.indices.as_slice()).into(),
            );

            list.set(index as u32, entry.into());
        }

        let ramps = Array::new_with_length(gradients.len() as u32);
        for (index, gradient) in gradients.iter().enumerate() {
            let entry = Object::new();
            set_f64(&entry, "kind", gradient.kind as f64);
            set_f64(&entry, "repeat", gradient.repeat as f64);
            set_f64(&entry, "interpolation", gradient.interpolation as f64);
            set_f64(&entry, "focalPoint", gradient.focal_point as f64);

            // Stops go across as two parallel arrays rather than an array of
            // objects: a gradient can have up to 15 stops and this is read once
            // per shape, but there is no reason to allocate per stop.
            let mut ratios = Vec::with_capacity(gradient.stops.len());
            let mut colors = Vec::with_capacity(gradient.stops.len() * 4);
            for (ratio, rgba) in &gradient.stops {
                ratios.push(*ratio);
                colors.extend_from_slice(rgba);
            }
            set(&entry, "ratios", &Float32Array::from(ratios.as_slice()).into());
            set(&entry, "colors", &Float32Array::from(colors.as_slice()).into());

            ramps.set(index as u32, entry.into());
        }

        let event = Object::new();
        set_f64(&event, "event", EVENT_SHAPE);
        set_f64(&event, "id", id as f64);
        set(&event, "draws", &list.into());
        set(&event, "gradients", &ramps.into());
        self.emit(&event);
    }

    fn register_bitmap(&mut self, id: BitmapId, width: u32, height: u32, rgba: Vec<u8>) {
        self.emit_bitmap(EVENT_BITMAP, id, width, height, rgba);
    }

    fn update_bitmap(&mut self, id: BitmapId, width: u32, height: u32, rgba: Vec<u8>) {
        self.emit_bitmap(EVENT_BITMAP_UPDATE, id, width, height, rgba);
    }

    fn submit_frame(
        &mut self,
        clear: [f32; 4],
        viewport: (u32, u32),
        commands: Vec<FrameCommand>,
    ) {
        if self.callback().is_none() {
            return;
        }

        let mut packed = vec![0.0f32; commands.len() * RECORD_STRIDE];
        for (index, command) in commands.iter().enumerate() {
            let slot = &mut packed[index * RECORD_STRIDE..(index + 1) * RECORD_STRIDE];
            match command {
                FrameCommand::DrawShape { shape, transform } => {
                    slot[0] = OP_DRAW_SHAPE;
                    slot[1] = *shape as f32;
                    write_transform(slot, transform);
                }
                FrameCommand::DrawBitmap {
                    bitmap,
                    transform,
                    smoothing,
                } => {
                    slot[0] = OP_DRAW_BITMAP;
                    slot[1] = *bitmap as f32;
                    write_transform(slot, transform);
                    slot[16] = if *smoothing { 1.0 } else { 0.0 };
                }
                FrameCommand::DrawRect { color, matrix } => {
                    slot[0] = OP_DRAW_RECT;
                    slot[2..8].copy_from_slice(matrix);
                    // A rect has no color transform, so the multiply slots
                    // carry its flat colour instead.
                    slot[8..12].copy_from_slice(color);
                }
                FrameCommand::PushMask => slot[0] = OP_PUSH_MASK,
                FrameCommand::ActivateMask => slot[0] = OP_ACTIVATE_MASK,
                FrameCommand::DeactivateMask => slot[0] = OP_DEACTIVATE_MASK,
                FrameCommand::PopMask => slot[0] = OP_POP_MASK,
                FrameCommand::BeginBlend { mode } => {
                    slot[0] = OP_BEGIN_BLEND;
                    slot[1] = blend_mode_id(*mode);
                }
                FrameCommand::EndBlend => slot[0] = OP_END_BLEND,
            }
        }

        let event = Object::new();
        set_f64(&event, "event", EVENT_FRAME);
        set_f64(&event, "count", commands.len() as f64);
        set_f64(&event, "width", viewport.0 as f64);
        set_f64(&event, "height", viewport.1 as f64);
        set_f64(&event, "stride", RECORD_STRIDE as f64);
        set(&event, "clear", &Float32Array::from(&clear[..]).into());
        set(
            &event,
            "records",
            &Float32Array::from(packed.as_slice()).into(),
        );
        self.emit(&event);
    }
}

impl JsCommandSink {
    fn emit_bitmap(&self, event_id: f64, id: BitmapId, width: u32, height: u32, rgba: Vec<u8>) {
        if self.callback().is_none() {
            return;
        }
        let event = Object::new();
        set_f64(&event, "event", event_id);
        set_f64(&event, "id", id as f64);
        set_f64(&event, "width", width as f64);
        set_f64(&event, "height", height as f64);
        set(&event, "rgba", &Uint8Array::from(rgba.as_slice()).into());
        self.emit(&event);
    }
}

fn write_transform(slot: &mut [f32], transform: &crate::sink::DrawTransform) {
    slot[2..8].copy_from_slice(&transform.matrix);
    slot[8..12].copy_from_slice(&transform.mult);
    slot[12..16].copy_from_slice(&transform.add);
}

/// Flatten a 3x3 into a 9-element Float32Array, row major.
fn flatten_matrix3(matrix: &[[f32; 3]; 3]) -> Float32Array {
    let flat: [f32; 9] = [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
    ];
    Float32Array::from(&flat[..])
}
