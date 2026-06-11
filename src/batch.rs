use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct BatchJob {
    pub request: crate::inference::AlignRequest,
    pub output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ManifestJob {
    audio: PathBuf,
    text: PathBuf,
    output: Option<PathBuf>,
    language: Option<String>,
}

pub fn load_manifest_jobs(
    manifest_path: &Path,
    output_dir: &Path,
    default_language: &str,
) -> Result<Vec<BatchJob>> {
    let manifest_text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest_dir = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut jobs = Vec::new();

    for (line_index, line) in manifest_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let raw: ManifestJob = serde_json::from_str(line).with_context(|| {
            format!("invalid manifest json at {}:{}", manifest_path.display(), line_index + 1)
        })?;
        let audio = resolve_relative_to(manifest_dir, raw.audio);
        let text = resolve_relative_to(manifest_dir, raw.text);
        let output = match raw.output {
            Some(output) => resolve_relative_to(output_dir, output),
            None => default_output_path(&audio, output_dir)?,
        };

        jobs.push(BatchJob {
            request: crate::inference::AlignRequest::new(
                crate::inference::AudioInput::Path(audio),
                crate::inference::TextInput::Path(text),
                raw.language
                    .filter(|language| !language.trim().is_empty())
                    .unwrap_or_else(|| default_language.to_string()),
            ),
            output,
        });
    }

    if jobs.is_empty() {
        anyhow::bail!("manifest has no jobs: {}", manifest_path.display());
    }
    Ok(jobs)
}

fn resolve_relative_to(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() { path } else { base.join(path) }
}

fn default_output_path(audio: &Path, output_dir: &Path) -> Result<PathBuf> {
    Ok(output_dir.join(format!("{}.json", file_stem_string(audio)?)))
}

fn file_stem_string(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(ToOwned::to_owned)
        .with_context(|| format!("path has no UTF-8 file stem: {}", path.display()))
}
