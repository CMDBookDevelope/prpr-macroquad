use crate::{get_context, get_quad_context, time::get_time};

use std::{cell::RefCell, collections::HashMap};

thread_local! {
    static PROFILER: RefCell<Profiler> = RefCell::new(Profiler::new());
}

fn with_profiler<R>(f: impl FnOnce(&mut Profiler) -> R) -> R {
    PROFILER.with(|profiler| f(&mut profiler.borrow_mut()))
}

impl Profiler {
    fn new() -> Self {
        Self {
            backend: None,
            frame: Frame::new(),
            queries: HashMap::new(),
            active_query: None,
            prev_frame: Frame::new(),
            enabled: false,
            enable_request: None,
            capture_request: false,
            capture: false,
            drawcalls: vec![],
            strings: vec![],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Zone {
    pub name: String,
    pub start_time: f64,
    pub duration: f64,
    pub children: Vec<Zone>,

    parent: *mut Zone,
}

impl Zone {
    fn clone(&self, parent: *mut Zone) -> Zone {
        Zone {
            name: self.name.clone(),
            start_time: self.start_time,
            duration: self.duration,
            children: self
                .children
                .iter()
                .map(|zone| zone.clone(self as *const _ as *mut _))
                .collect(),
            parent,
        }
    }
}

pub struct ZoneGuard {
    _marker: (),
}

impl ZoneGuard {
    pub fn new(name: &str) -> ZoneGuard {
        begin_zone(name);
        ZoneGuard { _marker: () }
    }
}

impl Drop for ZoneGuard {
    fn drop(&mut self) {
        end_zone();
    }
}

pub fn enable() {
    with_profiler(|profiler| profiler.enable_request = Some(true));
}

pub fn disable() {
    with_profiler(|profiler| profiler.enable_request = Some(false));
}

pub fn begin_zone(name: &str) {
    with_profiler(|profiler| {
        if profiler.enabled {
            profiler.begin_zone(name);
        }
    });
}

pub fn end_zone() {
    with_profiler(|profiler| {
        if profiler.enabled {
            profiler.end_zone();
        }
    });
}

fn begin_gpu_query_if_supported(name: &str) -> bool {
    with_profiler(|profiler| {
        if !profiler.enabled || !profiler.gpu_timers_supported() {
            return false;
        }
        profiler.begin_gpu_query(name);
        true
    })
}

pub fn begin_gpu_query(name: &str) {
    begin_gpu_query_if_supported(name);
}

pub fn end_gpu_query() {
    with_profiler(|profiler| {
        if profiler.gpu_timers_supported() {
            profiler.end_gpu_query();
        }
    });
}

pub(crate) struct GpuQueryGuard(bool);

impl GpuQueryGuard {
    pub(crate) fn begin(name: &str) -> Self {
        Self(begin_gpu_query_if_supported(name))
    }
}

impl Drop for GpuQueryGuard {
    fn drop(&mut self) {
        if self.0 {
            end_gpu_query();
        }
    }
}

pub(crate) fn set_backend(backend: miniquad::conf::RenderingBackend) {
    with_profiler(|profiler| {
        if profiler.backend != Some(backend) {
            profiler.active_query = None;
            profiler.queries.clear();
            profiler.backend = Some(backend);
        }
    });
}

pub(crate) fn reset_context() {
    with_profiler(|profiler| *profiler = Profiler::new());
}

/// Workaround to stop gl capture on debug rendering
#[doc(hidden)]
pub fn pause_gl_capture() {
    if with_profiler(|profiler| profiler.capture) {
        crate::get_context().gl.capture(false);
    }
}

/// Workaround to stop gl capture on debug rendering
pub fn resume_gl_capture() {
    if with_profiler(|profiler| profiler.capture) {
        crate::get_context().gl.capture(false);
    }
}

pub(crate) fn reset() {
    with_profiler(|profiler| {
        assert!(
            profiler.frame.active_zone.is_null(),
            "New frame started with unpaired begin/end zones."
        );

        profiler.frame.full_frame_time = crate::time::get_frame_time();
        std::mem::swap(&mut profiler.prev_frame, &mut profiler.frame);
        profiler.frame = Frame::new();

        if let Some(enable) = profiler.enable_request.take() {
            profiler.enabled = enable;
        }

        if profiler.capture {
            profiler.capture = false;
            crate::get_context().gl.capture(false);
        }

        if profiler.capture_request {
            profiler.drawcalls.clear();
            profiler.capture = true;
            crate::get_context().gl.capture(true);
            profiler.capture_request = false;
        }
    });
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub full_frame_time: f32,
    pub zones: Vec<Zone>,
    active_zone: *mut Zone,
}

impl Frame {
    fn new() -> Frame {
        Frame {
            full_frame_time: 0.0,
            zones: vec![],
            active_zone: std::ptr::null_mut(),
        }
    }

    pub fn try_clone(&self) -> Option<Frame> {
        if self.active_zone.is_null() == false {
            return None;
        }

        Some(Frame {
            full_frame_time: self.full_frame_time,
            zones: self
                .zones
                .iter()
                .map(|zone| zone.clone(std::ptr::null_mut()))
                .collect(),
            active_zone: std::ptr::null_mut(),
        })
    }
}

pub fn frame() -> Frame {
    with_profiler(|profiler| profiler.prev_frame.clone())
}

pub fn gpu_queries() -> Vec<(String, u64)> {
    with_profiler(|profiler| {
        profiler
            .queries
            .iter()
            .map(|(name, query)| (name.to_owned(), query.value))
            .collect()
    })
}

pub fn sample_gpu_queries() {
    with_profiler(|profiler| {
        for query in &mut profiler.queries {
            query.1.force_resume = true;
        }
    });
}

struct Profiler {
    backend: Option<miniquad::conf::RenderingBackend>,
    frame: Frame,
    prev_frame: Frame,
    queries: HashMap<String, GpuQuery>,
    active_query: Option<String>,
    enabled: bool,
    capture_request: bool,
    capture: bool,
    enable_request: Option<bool>,
    drawcalls: Vec<DrawCallTelemetry>,
    strings: Vec<String>,
}

impl Profiler {
    fn gpu_timers_supported(&self) -> bool {
        self.backend == Some(miniquad::conf::RenderingBackend::OpenGL)
            && miniquad::graphics::ElapsedQuery::is_supported()
    }

    fn begin_gpu_query(&mut self, name: &str) {
        assert!(
            self.active_query.is_none(),
            "Only one active query is allowed by OpenGL"
        );

        let name = name.to_string();
        let query = self
            .queries
            .entry(name.clone())
            .or_insert_with(|| GpuQuery {
                query: miniquad::graphics::ElapsedQuery::new(),
                in_progress: false,
                value: 0,
                force_resume: false,
            });
        self.active_query = Some(name);
        if query.force_resume {
            query.in_progress = true;
            query.query.begin_query();
        }
    }

    fn end_gpu_query(&mut self) -> bool {
        let Some(name) = self.active_query.take() else {
            return false;
        };
        let query = self.queries.get_mut(&name).unwrap();
        if query.in_progress {
            query.force_resume = false;
            query.in_progress = false;
            query.query.end_query();
        }
        if query.query.is_available() {
            query.value = query.query.get_result();
        }
        true
    }

    fn begin_zone(&mut self, name: &str) {
        let zones = if self.frame.active_zone.is_null() {
            &mut self.frame.zones
        } else {
            unsafe { &mut (&mut *self.frame.active_zone).children }
        };

        zones.push(Zone {
            name: name.to_string(),
            start_time: get_time(),
            duration: 0.0,
            parent: self.frame.active_zone,
            children: vec![],
        });
        self.frame.active_zone = zones.last_mut().unwrap() as _;
    }

    fn end_zone(&mut self) {
        assert!(
            self.frame.active_zone.is_null() == false,
            "end_zone called without begin_zone"
        );

        let start_time = unsafe { (&mut *self.frame.active_zone).start_time };
        let duration = get_time() - start_time;

        unsafe { (&mut *self.frame.active_zone).duration = duration };
        self.frame.active_zone = unsafe { (&mut *self.frame.active_zone).parent };
    }
}

pub struct GpuQuery {
    pub query: miniquad::graphics::ElapsedQuery,
    pub in_progress: bool,
    pub value: u64,
    pub force_resume: bool,
}

pub fn scene_allocated_memory() -> usize {
    use crate::experimental::scene;

    scene::allocated_memory()
}

/// ```skip
/// {
///    let _t = telemetry::LogTimeGuard::new("Atlas build time");
///     mq::texture::build_textures_atlas();
/// }
/// ```
/// Will add "Time query: Atlas build time, 0.5s" string to
/// `telemetry::strings()`
pub struct LogTimeGuard<'a> {
    name: &'a str,
    start_time: f64,
}

impl<'a> LogTimeGuard<'a> {
    pub fn new(name: &'a str) -> LogTimeGuard {
        LogTimeGuard {
            name,
            start_time: get_time(),
        }
    }
}

impl<'a> Drop for LogTimeGuard<'a> {
    fn drop(&mut self) {
        log_string(&format!(
            "Time query: {}, {:.1}s",
            self.name,
            get_time() - self.start_time
        ))
    }
}

pub fn log_string(string: &str) {
    with_profiler(|profiler| profiler.strings.push(string.to_owned()));
}

pub fn drawcalls() -> Vec<DrawCallTelemetry> {
    with_profiler(|profiler| profiler.drawcalls.clone())
}

pub fn strings() -> Vec<String> {
    with_profiler(|profiler| profiler.strings.clone())
}

/// Note that coroutines pre-allocate 56Kb, so 56000 as a result doesnt mean a leak
pub fn coroutines_allocated_memory() -> usize {
    get_context().coroutines_context.allocated_memory()
}

pub fn active_coroutines_count() -> usize {
    get_context().coroutines_context.active_coroutines_count()
}

pub fn capture_frame() {
    with_profiler(|profiler| profiler.capture_request = true);
}

#[derive(Clone, Debug)]
pub struct DrawCallTelemetry {
    pub indices_count: usize,
    pub texture: miniquad::Texture,
}

pub(crate) fn track_drawcall(
    pipeline: &miniquad::Pipeline,
    bindings: &miniquad::Bindings,
    indices_count: usize,
) {
    let texture = miniquad::Texture::new_render_texture(
        get_quad_context(),
        miniquad::TextureParams {
            width: 128,
            height: 128,
            ..Default::default()
        },
    );

    let pass = Some(miniquad::RenderPass::new(get_quad_context(), texture, None));
    get_quad_context().begin_pass(pass, miniquad::PassAction::clear_color(0.4, 0.8, 0.4, 1.));
    get_quad_context().apply_pipeline(pipeline);
    get_quad_context().apply_bindings(bindings);
    get_quad_context().draw(0, indices_count as _, 1);
    get_quad_context().end_render_pass();

    with_profiler(|profiler| {
        profiler.drawcalls.push(DrawCallTelemetry {
            indices_count,
            texture,
        });
    });
}
