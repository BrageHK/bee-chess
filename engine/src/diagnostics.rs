//! A small internal diagnostics abstraction, so engine/search code can
//! report what it's doing without knowing anything about UCI text
//! formatting or how (or whether) a diagnostic ends up visible.
//!
//! Per ADR 0001, no UCI strings may appear below the UCI adapter --
//! that includes diagnostics. Core/search code emits through
//! `Diagnostics::emit`; only `crate::uci` decides whether and how an
//! emitted diagnostic becomes an `info string ...` line (gated on
//! `debug on`/`off`), and it is also free to route diagnostics
//! elsewhere later (stderr, a log file, a telemetry channel) without
//! any call site here changing.
//!
//! This is deliberately not a general logging/tracing framework: no
//! per-node search logging belongs here (at millions of nodes/sec that
//! would destroy performance regardless of destination), and normal
//! search progress (depth, score, nodes, NPS, PV) belongs in real UCI
//! `info` fields, not `info string` diagnostics -- see `emit`'s docs.

/// How significant a diagnostic is. Mirrors common logging levels
/// rather than inventing new vocabulary; `Engine` does not currently
/// filter by level (all emitted diagnostics are equally visible when
/// debug mode is on), but call sites should still pick the level that
/// actually describes the message, since a future consumer (e.g.
/// routing only `Warn`/`Error` to stderr unconditionally, independent
/// of `debug`) can rely on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiagnosticLevel {
    Debug,
    Info,
    Warn,
    Error,
}

/// One emitted diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
}

/// Something that can receive diagnostics from engine/search code.
///
/// Use this for occasional, human-readable status messages: "position
/// set to startpos", "ignored unknown command: foo", "search
/// cancelled". Do **not** use it for per-node search progress, and do
/// not use it as a substitute for real UCI `info` fields -- structured
/// search telemetry (depth/score/nodes/nps/time/pv) has dedicated UCI
/// fields for exactly that purpose and should never be squeezed into
/// an `info string` instead.
pub trait Diagnostics {
    fn emit(&mut self, level: DiagnosticLevel, message: impl Into<String>);
}

/// A `Diagnostics` implementation that buffers emitted diagnostics in
/// memory until something drains them. `Engine` uses this: engine/
/// search code emits into it during a command's handling, and
/// `crate::uci::run` drains it afterward, deciding then (based on
/// `debug on`/`off`) whether each one becomes an `info string` line.
/// Buffering rather than emitting immediately keeps `Engine` itself
/// from needing to know whether debug mode is on, or what a UCI
/// adapter even is.
#[derive(Debug, Default)]
pub struct DiagnosticBuffer {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes and returns every diagnostic emitted since the last
    /// drain, in emission order.
    pub fn drain(&mut self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.diagnostics)
    }
}

impl Diagnostics for DiagnosticBuffer {
    fn emit(&mut self, level: DiagnosticLevel, message: impl Into<String>) {
        self.diagnostics.push(Diagnostic {
            level,
            message: message.into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer_drains_empty() {
        let mut buffer = DiagnosticBuffer::new();
        assert_eq!(buffer.drain(), Vec::new());
    }

    #[test]
    fn emit_then_drain_returns_in_order() {
        let mut buffer = DiagnosticBuffer::new();
        buffer.emit(DiagnosticLevel::Info, "first");
        buffer.emit(DiagnosticLevel::Warn, "second");

        assert_eq!(
            buffer.drain(),
            vec![
                Diagnostic {
                    level: DiagnosticLevel::Info,
                    message: "first".to_string()
                },
                Diagnostic {
                    level: DiagnosticLevel::Warn,
                    message: "second".to_string()
                },
            ]
        );
    }

    #[test]
    fn drain_clears_the_buffer() {
        let mut buffer = DiagnosticBuffer::new();
        buffer.emit(DiagnosticLevel::Debug, "one");
        buffer.drain();

        assert_eq!(buffer.drain(), Vec::new());
    }

    #[test]
    fn accepts_owned_and_borrowed_strings() {
        let mut buffer = DiagnosticBuffer::new();
        buffer.emit(DiagnosticLevel::Info, "borrowed");
        buffer.emit(DiagnosticLevel::Info, String::from("owned"));
        buffer.emit(DiagnosticLevel::Info, format!("formatted {}", 42));

        let messages: Vec<String> = buffer.drain().into_iter().map(|d| d.message).collect();
        assert_eq!(messages, vec!["borrowed", "owned", "formatted 42"]);
    }
}
