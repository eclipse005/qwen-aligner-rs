use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen forced aligner CLI (cudarc / CUDA).")]
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
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Align { audio, text, model, language, output } => {
            let aligner = qwen_aligner_rs::AlignerInference::load(&model)?;
            let result = aligner.align(qwen_aligner_rs::AlignRequest::from_paths(audio, text, language))?;
            qwen_aligner_rs::inference::write_forced_align_items_json(&output, &result.items)?;
            println!("align completed: words={}, output={}", result.items.len(), output.display());
        }
        Command::Batch { manifest, model, language, output_dir } => {
            let jobs = qwen_aligner_rs::batch::load_manifest_jobs(&manifest, &output_dir, &language)?;
            let aligner = qwen_aligner_rs::AlignerInference::load(&model)?;
            let results = aligner.align_batch(jobs.iter().map(|j| j.request.clone()))?;
            for (job, result) in jobs.iter().zip(results.iter()) {
                qwen_aligner_rs::inference::write_forced_align_items_json(&job.output, &result.items)
                    .with_context(|| format!("failed to write {}", job.output.display()))?;
            }
            println!("batch completed: jobs={}, output_dir={}", results.len(), output_dir.display());
        }
    }
    Ok(())
}
