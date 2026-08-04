//! Rendering for the Python-style path templates stored in `LeRobot` metadata.

use std::fmt;
use std::path::{Path, PathBuf};

/// A portable, non-empty path known to stay beneath a dataset root.
///
/// The stored representation always uses forward slashes. Absolute paths,
/// traversal components, empty components, Windows drive prefixes, and
/// backslashes are rejected at construction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DatasetRelativePath(String);

impl DatasetRelativePath {
    /// Parses an external path into the portable dataset-relative domain.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetRelativePathError`] when the path could escape a
    /// dataset root or cannot be represented portably.
    pub fn parse(path: impl Into<String>) -> Result<Self, DatasetRelativePathError> {
        let path = path.into();
        let first_component_has_windows_drive_prefix = path
            .as_bytes()
            .get(..2)
            .is_some_and(|prefix| prefix[0].is_ascii_alphabetic() && prefix[1] == b':');
        let components_are_normal = path
            .split('/')
            .all(|component| !component.is_empty() && component != "." && component != "..");
        if path.is_empty()
            || path.starts_with('/')
            || path.contains('\\')
            || first_component_has_windows_drive_prefix
            || !components_are_normal
        {
            return Err(DatasetRelativePathError { path });
        }
        Ok(Self(path))
    }

    /// Portable forward-slash representation used by storage providers.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Resolves this path beneath `dataset_root`.
    #[must_use]
    pub fn join_under(&self, dataset_root: &Path) -> PathBuf {
        dataset_root.join(&self.0)
    }

    /// Consumes the value and returns its portable representation.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for DatasetRelativePath {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for DatasetRelativePath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A path that is not safe to resolve beneath a dataset root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetRelativePathError {
    path: String,
}

impl DatasetRelativePathError {
    /// Returns the rejected path text.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl fmt::Display for DatasetRelativePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "dataset-relative path {:?} is not portable and safe",
            self.path
        )
    }
}

impl std::error::Error for DatasetRelativePathError {}

/// An invalid or incomplete `LeRobot` path template.
///
/// The error is intentionally opaque: callers report its context but do not
/// recover differently based on which part of a dataset path was malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathTemplateRenderError {
    detail: String,
}

impl PathTemplateRenderError {
    fn new(detail: String) -> Self {
        Self { detail }
    }

    /// Consumes the error and returns its human-readable detail.
    #[must_use]
    pub fn into_detail(self) -> String {
        self.detail
    }
}

impl fmt::Display for PathTemplateRenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for PathTemplateRenderError {}

/// Renders a `LeRobot` v3 path template such as
/// `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`.
///
/// Supported placeholders are `{video_key}`, `{chunk_index}`, and
/// `{file_index}`, each optionally with a zero-pad format spec like `:03d`.
///
/// # Errors
///
/// Returns [`PathTemplateRenderError`] when the template contains an unclosed
/// or unsupported placeholder, an unsupported format specifier, or requires a
/// video key that the caller did not provide.
pub fn render_path_template(
    template: &str,
    video_key: Option<&str>,
    chunk_index: u32,
    file_index: u32,
) -> Result<DatasetRelativePath, PathTemplateRenderError> {
    let mut rendered = String::with_capacity(template.len());
    let mut remaining = template;

    while let Some(open_position) = remaining.find('{') {
        rendered.push_str(&remaining[..open_position]);
        let after_open = &remaining[open_position + 1..];
        let Some(close_position) = after_open.find('}') else {
            return Err(PathTemplateRenderError::new(format!(
                "unclosed placeholder in path template {template:?}"
            )));
        };
        let placeholder = &after_open[..close_position];
        let (name, format_spec) = match placeholder.split_once(':') {
            Some((name, format_spec)) => (name, Some(format_spec)),
            None => (placeholder, None),
        };
        let substitution = match name {
            "video_key" => video_key
                .ok_or_else(|| {
                    PathTemplateRenderError::new(format!(
                        "path template {template:?} requires a video key but none applies here"
                    ))
                })?
                .to_owned(),
            "chunk_index" => render_index_placeholder(chunk_index, format_spec, template)?,
            "file_index" => render_index_placeholder(file_index, format_spec, template)?,
            unknown_name => {
                return Err(PathTemplateRenderError::new(format!(
                    "unsupported placeholder {{{unknown_name}}} in path template {template:?}"
                )));
            }
        };
        rendered.push_str(&substitution);
        remaining = &after_open[close_position + 1..];
    }

    rendered.push_str(remaining);
    DatasetRelativePath::parse(rendered).map_err(|source| {
        PathTemplateRenderError::new(format!(
            "path template {template:?} rendered an unsafe path: {source}"
        ))
    })
}

fn render_index_placeholder(
    value: u32,
    format_spec: Option<&str>,
    template: &str,
) -> Result<String, PathTemplateRenderError> {
    match format_spec {
        None | Some("d") => Ok(value.to_string()),
        Some(spec) if spec.starts_with('0') && spec.ends_with('d') && spec.len() > 2 => {
            let width: usize = spec[1..spec.len() - 1].parse().map_err(|_| {
                PathTemplateRenderError::new(format!(
                    "unsupported format spec {spec:?} in path template {template:?}"
                ))
            })?;
            Ok(format!("{value:0width$}"))
        }
        Some(unsupported_spec) => Err(PathTemplateRenderError::new(format!(
            "unsupported format spec {unsupported_spec:?} in path template {template:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_data_path_template() {
        let rendered = render_path_template(
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            None,
            0,
            0,
        )
        .expect("data path template renders");
        assert_eq!(rendered.as_str(), "data/chunk-000/file-000.parquet");
    }

    #[test]
    fn renders_video_path_template_with_key() {
        let rendered = render_path_template(
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            Some("observation.images.up"),
            12,
            3,
        )
        .expect("video path template renders");
        assert_eq!(
            rendered.as_str(),
            "videos/observation.images.up/chunk-012/file-003.mp4"
        );
    }

    #[test]
    fn index_wider_than_pad_is_not_truncated() {
        let rendered = render_path_template("chunk-{chunk_index:03d}", None, 12345, 0)
            .expect("wide index renders");
        assert_eq!(rendered.as_str(), "chunk-12345");
    }

    #[test]
    fn unpadded_placeholder_is_supported() {
        let rendered =
            render_path_template("file-{file_index}", None, 0, 7).expect("unpadded index renders");
        assert_eq!(rendered.as_str(), "file-7");
    }

    #[test]
    fn unknown_placeholder_is_rejected() {
        let error = render_path_template("data/{episode_index:06d}.parquet", None, 0, 0)
            .expect_err("unknown placeholder must be rejected");
        assert!(error.to_string().contains("episode_index"));
    }

    #[test]
    fn video_key_placeholder_without_key_is_rejected() {
        let error = render_path_template("videos/{video_key}/f.mp4", None, 0, 0)
            .expect_err("missing video key must be rejected");
        assert!(error.to_string().contains("requires a video key"));
    }

    #[test]
    fn unclosed_placeholder_is_rejected() {
        let error = render_path_template("data/{chunk_index", None, 0, 0)
            .expect_err("unclosed placeholder must be rejected");
        assert!(error.to_string().contains("unclosed placeholder"));
    }

    #[test]
    fn unsupported_format_spec_is_rejected() {
        let error = render_path_template("data/{chunk_index:.2f}", None, 0, 0)
            .expect_err("unsupported format must be rejected");
        assert!(error.to_string().contains(".2f"));
    }

    #[test]
    fn rendered_paths_cannot_escape_or_depend_on_platform_separators() {
        for unsafe_template in [
            "/tmp/file-{file_index}.parquet",
            "../file-{file_index}.parquet",
            "data//file-{file_index}.parquet",
            "C:/data/file-{file_index}.parquet",
            "data\\file-{file_index}.parquet",
        ] {
            assert!(
                render_path_template(unsafe_template, None, 0, 0).is_err(),
                "template should be rejected: {unsafe_template}"
            );
        }
    }
}
