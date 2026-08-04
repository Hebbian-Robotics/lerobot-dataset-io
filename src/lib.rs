#![doc = include_str!("../README.md")]

mod arrow_columns;
mod dataset;
mod error;
mod export;
pub mod format_extension_support;
mod info;
#[cfg(feature = "object-store")]
mod object_store_provider;
mod path_template;
mod source;
mod time;
mod vector_columns;

use std::fmt::Debug;
use std::path::Path;

use async_trait::async_trait;

pub use error::{ExportDestinationProblem, LerobotError};
pub use export::{
    CameraVideoWriteResult, DatasetExportExtension, EXPORT_STAGING_OWNERSHIP_MANIFEST_FILE_NAME,
    ExportEpisodes, ExportRangePlacement, ExportReport, InvalidWrittenVideo, PreparedExportInputs,
    SourceSegment, VideoSegmentWriter, WrittenVideo, export_episode_range_preserving_packed_videos,
    export_subset, export_subset_preserving_packed_videos,
    export_subset_preserving_packed_videos_with_extension, export_subset_with_extension,
    write_packed_subset_info_json, write_packed_subset_info_json_with_extension,
};
pub use info::{DatasetInfo, DatasetInfoParseError, FeatureDtype, FeatureSpec, VideoFeatureKind};
#[cfg(feature = "object-store")]
pub use object_store_provider::{ObjectStoreDatasetFileProvider, ObjectStoreDatasetSourceError};
pub use path_template::{
    DatasetRelativePath, DatasetRelativePathError, PathTemplateRenderError, render_path_template,
};
pub use source::{DatasetCachePolicy, DatasetSource, RemoteDatasetSource};
pub use time::{FramesPerSecond, InvalidFramesPerSecond, InvalidTimestampRange, TimestampRange};
pub use vector_columns::{
    EpisodeDataFileReadPlan, EpisodeDataReadPlan, EpisodeVectorColumn, EpisodeVectorColumns,
    EpisodeVectorReadReport, EpisodeVectorReadSchema, EpisodeVectorRow, EpisodeVectorRowReader,
    EpisodeVectorSelectionMode, InvalidEpisodeDataReadPlan, PlannedEpisodeDataRows,
    read_episodes_vector_columns_from_file,
};

/// Opaque failure returned by a [`DatasetFileProvider`].
///
/// Dataset readers add operation, path, and source-identity context before
/// exposing the failure. Callers never branch on provider-specific failures,
/// so implementations can preserve their native error without forcing it into
/// a shared enum.
pub type DatasetFileProviderError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Result of materializing one remote dataset file into a local cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFileMaterialization {
    /// The file existed and is atomically available at the requested path.
    Materialized,
    /// The provider has no file at the requested dataset-relative path.
    NotFound,
}

/// The remote file capabilities required by a `LeRobot` dataset reader.
///
/// Implementations may wrap an object store, an HTTP service, or another
/// backend. Returned file names must be relative to the dataset root and use
/// forward slashes. Materialization must not leave a partial file at
/// `destination_path` when it fails.
#[async_trait]
pub trait DatasetFileProvider: Debug + Send + Sync {
    /// Lists every dataset-relative file beneath `relative_prefix`.
    ///
    /// `report_files_discovered` receives the cumulative number of files
    /// observed while the backend listing advances, before the complete list
    /// is available. The reader normalizes ordering after this boundary.
    async fn list_files(
        &self,
        relative_prefix: &str,
        report_files_discovered: &mut (dyn FnMut(u64) + Send),
    ) -> Result<Vec<String>, DatasetFileProviderError>;

    /// Atomically materializes `relative_path` into `destination_path`.
    async fn materialize_file(
        &self,
        relative_path: &str,
        destination_path: &Path,
    ) -> Result<DatasetFileMaterialization, DatasetFileProviderError>;
}

pub use dataset::{
    Dataset, DatasetOpenProgress, DatasetOpenProgressReporter, Episode, VideoSegment,
};
