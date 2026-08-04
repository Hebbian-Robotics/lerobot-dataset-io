use std::fmt;
use std::path::PathBuf;

/// Why the export destination guard rejected a destination directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExportDestinationProblem {
    /// The destination resolves to the source dataset root itself.
    SameAsSource,
    /// The destination lies inside the source dataset.
    InsideSource,
    /// The source dataset lies inside the destination.
    ContainsSource,
    /// The destination exists but is not a directory.
    NotADirectory,
    /// The destination directory already contains entries.
    NotEmpty,
    /// Another export currently holds the destination's claim marker.
    AlreadyClaimed,
}

impl fmt::Display for ExportDestinationProblem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            Self::SameAsSource => "destination is the source dataset itself",
            Self::InsideSource => "destination is inside the source dataset",
            Self::ContainsSource => "destination contains the source dataset",
            Self::NotADirectory => "destination exists and is not a directory",
            Self::NotEmpty => "destination directory is not empty",
            Self::AlreadyClaimed => "another export is already writing to this destination",
        };
        formatter.write_str(description)
    }
}

/// Errors returned by the `LeRobot` v3 dataset reader and export writer.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LerobotError {
    /// A required local filesystem operation failed.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// A required JSON document could not be decoded or encoded.
    #[error("JSON error in {context}: {source}")]
    Json {
        /// Human-readable name of the document being processed.
        context: String,
        /// Underlying JSON codec error.
        #[source]
        source: serde_json::Error,
    },

    /// A packed parquet file could not be opened or decoded.
    #[error("parquet error at {path}: {source}")]
    Parquet {
        /// Path of the parquet file being processed.
        path: PathBuf,
        /// Underlying parquet reader or writer error.
        #[source]
        source: parquet::errors::ParquetError,
    },

    /// An Arrow array or schema operation failed.
    #[error("arrow error: {source}")]
    Arrow {
        /// Underlying Arrow error.
        #[from]
        source: arrow_schema::ArrowError,
    },

    /// A required column is absent from a packed parquet schema.
    #[error("missing required column {name:?}")]
    MissingColumn {
        /// Exact missing column name.
        name: String,
    },

    /// The dataset declares an unsupported `LeRobot` format version.
    #[error("unsupported LeRobot codebase version {found:?} (this reader supports v3.x)")]
    UnsupportedVersion {
        /// Version string declared by the dataset.
        found: String,
    },

    /// Dataset metadata contradicts itself or the packed file contents.
    #[error("inconsistent metadata: {detail}")]
    InconsistentMetadata {
        /// Description of the violated format invariant.
        detail: String,
    },

    /// A remote source could not list or materialize a required object.
    #[error("remote dataset source error while {context}: {detail}")]
    Source {
        /// Operation being attempted when the source failed.
        context: String,
        /// Provider-neutral source failure description.
        detail: String,
    },

    /// A requested episode index is not declared by the dataset.
    #[error("episode {index} not found")]
    EpisodeNotFound {
        /// Missing episode index.
        index: u32,
    },

    /// A synchronous read was attempted before a remote object was cached.
    #[error(
        "object {relative_key:?} of a remote dataset is not materialized locally; \
         await the async ensure/prefetch methods before synchronous reads"
    )]
    NotMaterialized {
        /// Dataset-relative key that must be materialized first.
        relative_key: String,
    },

    /// The destination guard rejected an unsafe or occupied export path.
    #[error("refusing export destination {dest}: {problem}")]
    ExportDestination {
        /// Rejected destination path.
        dest: PathBuf,
        /// Semantic reason the destination is unsafe to claim.
        problem: ExportDestinationProblem,
    },

    /// The caller-provided video writer failed while exporting one camera.
    #[error("video segment writer failed for camera {camera:?}: {source}")]
    VideoWrite {
        /// Camera feature being exported.
        camera: String,
        /// Error returned by the caller-provided writer.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl From<crate::PathTemplateRenderError> for LerobotError {
    fn from(source: crate::PathTemplateRenderError) -> Self {
        Self::InconsistentMetadata {
            detail: source.into_detail(),
        }
    }
}

impl From<crate::DatasetInfoParseError> for LerobotError {
    fn from(source: crate::DatasetInfoParseError) -> Self {
        match source {
            crate::DatasetInfoParseError::Json { source } => Self::Json {
                context: "info.json".to_owned(),
                source,
            },
            crate::DatasetInfoParseError::UnsupportedVersion { found } => {
                Self::UnsupportedVersion { found }
            }
            crate::DatasetInfoParseError::InvalidFramesPerSecond { source } => {
                Self::InconsistentMetadata {
                    detail: source.to_string(),
                }
            }
            crate::DatasetInfoParseError::InvalidPathTemplate { field, source } => {
                Self::InconsistentMetadata {
                    detail: format!("invalid {field} in info.json: {source}"),
                }
            }
        }
    }
}

impl From<crate::InvalidEpisodeDataReadPlan> for LerobotError {
    fn from(source: crate::InvalidEpisodeDataReadPlan) -> Self {
        Self::InconsistentMetadata {
            detail: source.to_string(),
        }
    }
}
