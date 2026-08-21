//! The command stream is the contract between the fork and the three.js host,
//! so the flattening of Ruffle's nested `CommandList` into a flat draw list is
//! worth pinning down before anything consumes it.

use std::cell::RefCell;
use std::rc::Rc;

use ruffle_render::backend::RenderBackend;
use ruffle_render::commands::{Command, CommandList};
use ruffle_render::matrix::Matrix;
use ruffle_render_stream::{CommandSink, FrameCommand, ShapeDraw, StreamRenderBackend};

/// A sink that appends every frame into a shared buffer, so a test can inspect
/// what the backend produced after driving it.
#[derive(Default)]
struct SharedSink {
    frames: Rc<RefCell<Vec<Vec<FrameCommand>>>>,
}

impl CommandSink for SharedSink {
    fn register_shape(
        &mut self,
        _id: u32,
        _draws: Vec<ShapeDraw>,
        _gradients: Vec<ruffle_render_stream::GradientSpec>,
    ) {
    }
    fn register_bitmap(&mut self, _id: u32, _w: u32, _h: u32, _rgba: Vec<u8>) {}
    fn update_bitmap(&mut self, _id: u32, _w: u32, _h: u32, _rgba: Vec<u8>) {}
    fn submit_frame(
        &mut self,
        _clear: [f32; 4],
        _viewport: (u32, u32),
        commands: Vec<FrameCommand>,
    ) {
        self.frames.borrow_mut().push(commands);
    }
}

fn backend_with_sink() -> (StreamRenderBackend, Rc<RefCell<Vec<Vec<FrameCommand>>>>) {
    let frames = Rc::new(RefCell::new(Vec::new()));
    let sink = SharedSink {
        frames: Rc::clone(&frames),
    };
    (StreamRenderBackend::new(Box::new(sink)), frames)
}

fn rect(r: u8) -> Command {
    Command::DrawRect {
        color: swf::Color {
            r,
            g: 0,
            b: 0,
            a: 255,
        },
        matrix: Matrix::IDENTITY,
    }
}

#[test]
fn submits_one_frame_per_call() {
    let (mut backend, frames) = backend_with_sink();

    backend.submit_frame(swf::Color::BLACK, CommandList::default(), Vec::new());
    backend.submit_frame(swf::Color::BLACK, CommandList::default(), Vec::new());

    assert_eq!(frames.borrow().len(), 2);
}

#[test]
fn flattens_draw_rect() {
    let (mut backend, frames) = backend_with_sink();

    let mut list = CommandList::default();
    list.commands.push(rect(1));
    list.commands.push(rect(2));
    backend.submit_frame(swf::Color::BLACK, list, Vec::new());

    let frames = frames.borrow();
    assert_eq!(frames[0].len(), 2);
    assert!(matches!(frames[0][0], FrameCommand::DrawRect { .. }));
}

#[test]
fn alpha_mask_becomes_flat_marker_sequence() {
    let (mut backend, frames) = backend_with_sink();

    let mut mask = CommandList::default();
    mask.commands.push(rect(1));

    let mut maskee = CommandList::default();
    maskee.commands.push(rect(2));
    maskee.commands.push(rect(3));

    let mut list = CommandList::default();
    list.commands.push(Command::RenderAlphaMask {
        maskee_commands: maskee,
        mask_commands: mask,
    });
    backend.submit_frame(swf::Color::BLACK, list, Vec::new());

    let frames = frames.borrow();
    let frame = &frames[0];

    // The nested tree must arrive as: push, <mask draws>, activate,
    // <maskee draws>, deactivate, pop.
    assert!(matches!(frame[0], FrameCommand::PushMask));
    assert!(matches!(frame[1], FrameCommand::DrawRect { .. }));
    assert!(matches!(frame[2], FrameCommand::ActivateMask));
    assert!(matches!(frame[3], FrameCommand::DrawRect { .. }));
    assert!(matches!(frame[4], FrameCommand::DrawRect { .. }));
    assert!(matches!(frame[5], FrameCommand::DeactivateMask));
    assert!(matches!(frame[6], FrameCommand::PopMask));
    assert_eq!(frame.len(), 7);
}

#[test]
fn blend_wraps_its_children_in_markers() {
    let (mut backend, frames) = backend_with_sink();

    let mut inner = CommandList::default();
    inner.commands.push(rect(1));

    let mut list = CommandList::default();
    list.commands.push(Command::Blend(
        inner,
        ruffle_render::commands::RenderBlendMode::Builtin(swf::BlendMode::Multiply),
    ));
    backend.submit_frame(swf::Color::BLACK, list, Vec::new());

    let frames = frames.borrow();
    let frame = &frames[0];

    match &frame[0] {
        FrameCommand::BeginBlend { mode } => assert_eq!(*mode, swf::BlendMode::Multiply),
        other => panic!("expected BeginBlend, got {other:?}"),
    }
    assert!(matches!(frame[1], FrameCommand::DrawRect { .. }));
    assert!(matches!(frame[2], FrameCommand::EndBlend));
}

#[test]
fn stage3d_and_debug_lines_are_dropped() {
    let (mut backend, frames) = backend_with_sink();

    let mut list = CommandList::default();
    list.commands.push(Command::DrawLine {
        color: swf::Color::BLACK,
        matrix: Matrix::IDENTITY,
    });
    list.commands.push(Command::DrawLineRect {
        color: swf::Color::BLACK,
        matrix: Matrix::IDENTITY,
    });
    list.commands.push(rect(1));
    backend.submit_frame(swf::Color::BLACK, list, Vec::new());

    let frames = frames.borrow();
    // Only the rect survives; the debug-overlay primitives are dropped.
    assert_eq!(frames[0].len(), 1);
}

#[test]
fn declares_itself_unsupported_for_offscreen_work() {
    let (backend, _) = backend_with_sink();
    assert!(!backend.is_offscreen_supported());
    assert_eq!(backend.name(), "stream");
}
