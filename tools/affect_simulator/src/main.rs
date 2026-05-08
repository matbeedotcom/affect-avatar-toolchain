//! `affect-sim` CLI.
//!
//! Subcommands:
//!     affect-sim run <scenario.json> [--out trace.json]
//!     affect-sim plot <trace.json>
//!     affect-sim run-all <dir/>

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use affect_simulator::runner::{load_scenario, write_trace, SimulatorRun, Trace, TraceFrame};

#[derive(Parser, Debug)]
#[command(name = "affect-sim", about = "Affect-state simulator (Phase 1)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run a scenario file end-to-end, optionally writing the trace.
    Run {
        scenario: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Print the per-tick prompt summary to stdout (verbose).
        #[arg(long, default_value_t = false)]
        verbose: bool,
    },
    /// Render an ASCII plot of channels over time from a trace JSON.
    Plot {
        trace: PathBuf,
        /// Maximum width of each channel's plot row in chars.
        #[arg(long, default_value_t = 60)]
        width: usize,
    },
    /// Run every `*.json` scenario in a directory; print pass/fail summary.
    RunAll {
        dir: PathBuf,
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run {
            scenario,
            out,
            verbose,
        } => cmd_run(&scenario, out.as_deref(), verbose),
        Cmd::Plot { trace, width } => cmd_plot(&trace, width),
        Cmd::RunAll { dir, out_dir } => cmd_run_all(&dir, out_dir.as_deref()),
    }
}

fn cmd_run(scenario_path: &Path, out: Option<&Path>, verbose: bool) -> Result<()> {
    let scenario = load_scenario(scenario_path)?;
    let run = SimulatorRun::default();
    let trace = run.run(&scenario);
    print_summary(&trace);
    if verbose {
        for frame in &trace.frames {
            if !frame.events.is_empty() {
                println!("\n--- t={} ms ---", frame.timestamp_ms);
                for e in &frame.events {
                    println!("  event: {:?} ({})", e.kind, e.id);
                }
                print!("{}", frame.prompt);
            }
        }
    }
    if let Some(path) = out {
        write_trace(path, &trace)?;
        println!("trace written: {}", path.display());
    }
    Ok(())
}

fn cmd_plot(trace_path: &Path, width: usize) -> Result<()> {
    let bytes = std::fs::read(trace_path)
        .with_context(|| format!("reading trace: {}", trace_path.display()))?;
    let trace: Trace = serde_json::from_slice(&bytes)?;
    print_summary(&trace);
    println!();
    plot_channels(&trace.frames, width);
    plot_core(&trace.frames, width);
    Ok(())
}

fn cmd_run_all(dir: &Path, out_dir: Option<&Path>) -> Result<()> {
    if let Some(out) = out_dir {
        std::fs::create_dir_all(out)?;
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading scenarios dir: {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort();

    println!(
        "running {} scenarios from {}",
        entries.len(),
        dir.display()
    );
    for path in entries {
        let scenario = load_scenario(&path)?;
        let run = SimulatorRun::default();
        let trace = run.run(&scenario);
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        println!("\n[{}] {}", stem, scenario.description);
        print_summary(&trace);
        if let Some(out) = out_dir {
            let out_path = out.join(format!("{}.trace.json", stem));
            write_trace(&out_path, &trace)?;
        }
    }
    Ok(())
}

fn print_summary(trace: &Trace) {
    let last = match trace.frames.last() {
        Some(f) => f,
        None => {
            println!("(empty trace)");
            return;
        }
    };
    let max_arousal = trace
        .frames
        .iter()
        .map(|f| f.core.arousal)
        .fold(0.0_f32, f32::max);
    let min_valence = trace
        .frames
        .iter()
        .map(|f| f.core.valence)
        .fold(1.0_f32, f32::min);
    let max_anger = trace
        .frames
        .iter()
        .map(|f| f.channels.anger)
        .fold(0.0_f32, f32::max);
    let event_count: usize = trace.frames.iter().map(|f| f.events.len()).sum();
    println!(
        "  {} ticks, {} events; final V/A/D = {:+.2} / {:.2} / {:+.2}; peak arousal {:.2}, lowest valence {:+.2}, peak anger {:.2}",
        trace.frames.len(),
        event_count,
        last.core.valence,
        last.core.arousal,
        last.core.dominance,
        max_arousal,
        min_valence,
        max_anger,
    );
}

fn plot_channels(frames: &[TraceFrame], width: usize) {
    println!("Channels (rows = channel, cols = time):");
    let names: [(&str, fn(&TraceFrame) -> f32); 8] = [
        ("anger      ", |f| f.channels.anger),
        ("sadness    ", |f| f.channels.sadness),
        ("fear       ", |f| f.channels.fear),
        ("joy        ", |f| f.channels.joy),
        ("calm       ", |f| f.channels.calm),
        ("frustration", |f| f.channels.frustration),
        ("curiosity  ", |f| f.channels.curiosity),
        ("empathy    ", |f| f.channels.empathy),
    ];
    for (name, accessor) in names {
        let row = ascii_row(frames, width, accessor, 0.0, 1.0);
        println!("  {} |{}|", name, row);
    }
}

fn plot_core(frames: &[TraceFrame], width: usize) {
    println!("Core (V in [-1,1], A in [0,1], D in [-1,1]):");
    let v = ascii_row(frames, width, |f| f.core.valence, -1.0, 1.0);
    let a = ascii_row(frames, width, |f| f.core.arousal, 0.0, 1.0);
    let d = ascii_row(frames, width, |f| f.core.dominance, -1.0, 1.0);
    println!("  valence     |{}|", v);
    println!("  arousal     |{}|", a);
    println!("  dominance   |{}|", d);
}

/// Down-sample frames to `width` columns and render with 5-level ramp.
fn ascii_row(
    frames: &[TraceFrame],
    width: usize,
    accessor: fn(&TraceFrame) -> f32,
    lo: f32,
    hi: f32,
) -> String {
    let n = frames.len().max(1);
    let mut out = String::with_capacity(width);
    let ramp = [' ', '.', ':', '+', '#'];
    for col in 0..width {
        let i_start = col * n / width;
        let i_end = (((col + 1) * n) / width).max(i_start + 1).min(n);
        let mut sum = 0.0;
        let mut count = 0;
        for i in i_start..i_end {
            sum += accessor(&frames[i]);
            count += 1;
        }
        let mean = if count == 0 { lo } else { sum / count as f32 };
        let norm = ((mean - lo) / (hi - lo)).clamp(0.0, 1.0);
        let idx = ((norm * (ramp.len() as f32 - 1.0)).round() as usize).min(ramp.len() - 1);
        out.push(ramp[idx]);
    }
    out
}
