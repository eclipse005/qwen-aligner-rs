use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use qwen_forced_aligner_rs::{
    load_model, write_forced_align_items_json, AlignRequest, DeviceRequest, ModelOptions,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen forced aligner CLI (cudarc / CPU).")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    #[command(about = "Align one audio file with one text file.")]
    Align {
        #[arg(long, value_name = "WAV", help = "Input .wav audio file.")]
        audio: PathBuf,
        #[arg(long, value_name = "TXT", help = "Input transcript text file.")]
        text: PathBuf,
        #[arg(long, value_name = "DIR", help = "Model directory.")]
        model: PathBuf,
        #[arg(long, default_value = "English", help = "Language name.")]
        language: String,
        #[arg(long, value_name = "JSON", help = "Output alignment JSON file.")]
        output: PathBuf,
        #[arg(
            long,
            default_value = "auto",
            help = "Backend selection: auto | cuda[:<ordinal>] | cpu"
        )]
        device: String,
    },
    #[command(about = "Batch mode: run jobs from a manifest JSONL file.")]
    Batch {
        #[arg(long, value_name = "JSONL", help = "Manifest JSONL file.")]
        manifest: PathBuf,
        #[arg(long, value_name = "DIR", help = "Model directory.")]
        model: PathBuf,
        #[arg(long, default_value = "English", help = "Default language.")]
        language: String,
        #[arg(long, value_name = "DIR", help = "Output directory for JSON files.")]
        output_dir: PathBuf,
        #[arg(long, default_value = "auto", help = "Backend selection.")]
        device: String,
    },
}

fn parse_device(s: &str) -> Result<DeviceRequest> {
    let s = s.trim().to_ascii_lowercase();
    if s == "auto" {
        return Ok(DeviceRequest::Auto);
    }
    if s == "cpu" {
        return Ok(DeviceRequest::Cpu);
    }
    if s == "cuda" {
        return Ok(DeviceRequest::Cuda(0));
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let n: usize = rest.parse().with_context(|| format!("invalid CUDA ordinal {:?}", rest))?;
        return Ok(DeviceRequest::Cuda(n));
    }
    anyhow::bail!("unknown --device {:?} (expected: auto | cuda | cuda:<n> | cpu)", s)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Align { audio, text, model, language, output, device } => {
            let opts = ModelOptions { device: parse_device(&device)? };
            let aligner = load_model(&model, opts)?;
            let result = aligner.align(AlignRequest::from_paths(audio, text, language))?;
            write_forced_align_items_json(&output, &result.items)?;
            println!("align completed: words={}, output={}", result.items.len(), output.display());
        }
        Command::Batch { manifest, model, language, output_dir, device } => {
            let jobs = qwen_forced_aligner_rs::load_manifest_jobs(&manifest, &output_dir, &language)?;
            let opts = ModelOptions { device: parse_device(&device)? };
            let aligner = load_model(&model, opts)?;
            let results = aligner.align_batch(jobs.iter().map(|j| j.request.clone()))?;
            for (job, result) in jobs.iter().zip(results.iter()) {
                write_forced_align_items_json(&job.output, &result.items)
                    .with_context(|| format!("failed to write {}", job.output.display()))?;
            }
            println!("batch completed: jobs={}, output_dir={}", results.len(), output_dir.display());
        }
    }
    Ok(())
}
