//! Affect-state simulator library.
//!
//! Implements Phase 1 of the activation-steering plan: a deterministic
//! pipeline of `PerceivedEvent → Appraisal → Dynamics → Regulation →
//! AffectState` that runs without an LLM. See
//! `docs/references/activation-steering-audio-llm/PROJECT_PLAN.md` §6 Phase 1.
//!
//! Modules are kept small and stateless where possible so the appraisal,
//! dynamics, and regulation logic can be lifted into runtime nodes during
//! Phase 2 without re-authoring the equations.

pub mod appraisal;
pub mod channel_a;
pub mod channel_d;
pub mod dynamics;
pub mod event;
pub mod regulation;
pub mod renderer;
pub mod runner;
pub mod state;

pub use appraisal::{Appraisal, AppraisalEngine};
pub use channel_a::{ChannelA, SamplingOverride};
pub use channel_d::compute_target as channel_d_target;
pub use dynamics::{ChannelParams, Dynamics, DynamicsParams};
pub use event::{EventKind, Modality, PerceivedEvent};
pub use regulation::{RegulationEngine, RegulationPolicy};
pub use renderer::PromptRenderer;
pub use runner::{Scenario, SimulatorRun, Trace, TraceFrame};
pub use state::{AffectState, CoreAffect, EmotionChannels, EventLogEntry};
