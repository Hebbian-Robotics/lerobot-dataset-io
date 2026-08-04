use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema, SchemaRef};
use nonempty::NonEmpty;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use serde_json::Value;

use crate::arrow_columns::{
    int64_row_value, require_field_position, require_int64_column, require_string_column,
};
use crate::dataset::{enumerate_chunked_parquet_files, open_parquet_reader_builder};
use crate::{
    Dataset, Episode, ExportDestinationProblem, LerobotError, TimestampRange, render_path_template,
};

/// One episode's slice of a source video file, handed to a
/// [`VideoSegmentWriter`] for concatenation into an export video.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSegment {
    /// Absolute path of the source video file.
    file: PathBuf,
    timestamp_range: TimestampRange,
}

impl SourceSegment {
    /// Creates a source-file segment from an already validated time range.
    #[must_use]
    pub fn new(file: PathBuf, timestamp_range: TimestampRange) -> Self {
        Self {
            file,
            timestamp_range,
        }
    }

    /// Absolute source video path.
    #[must_use]
    pub fn file(&self) -> &Path {
        &self.file
    }

    /// Inclusive-start, exclusive-end range within the source file.
    #[must_use]
    pub const fn timestamp_range(&self) -> TimestampRange {
        self.timestamp_range
    }
}

/// What a [`VideoSegmentWriter`] actually encoded into a destination file.
/// [`export_subset`] copies these facts into the exported `info.json` video
/// features, replacing the source declarations (every export re-encodes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenVideo {
    /// Codec name as `info.json` declares it, e.g. `"h264"`.
    codec: String,
    /// Width of the written video in pixels.
    width: u32,
    /// Height of the written video in pixels.
    height: u32,
}

impl WrittenVideo {
    /// Parses the facts reported by a video writer.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidWrittenVideo`] for a blank codec or zero dimension.
    pub fn new(
        codec: impl Into<String>,
        width: u32,
        height: u32,
    ) -> Result<Self, InvalidWrittenVideo> {
        let codec = codec.into();
        if codec.trim().is_empty() || width == 0 || height == 0 {
            return Err(InvalidWrittenVideo {
                codec,
                width,
                height,
            });
        }
        Ok(Self {
            codec,
            width,
            height,
        })
    }

    /// Codec name written to `info.json`.
    #[must_use]
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// Encoded width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Encoded height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
}

/// Video facts that cannot describe a written media file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidWrittenVideo {
    codec: String,
    width: u32,
    height: u32,
}

impl std::fmt::Display for InvalidWrittenVideo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "written video requires a non-blank codec and non-zero dimensions, found codec {:?} at {}x{}",
            self.codec, self.width, self.height
        )
    }
}

impl std::error::Error for InvalidWrittenVideo {}

/// Everything a [`VideoSegmentWriter`] reports about one written camera video.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraVideoWriteResult {
    /// For each input segment, in the same order, the timestamp range it
    /// occupies in the destination file.
    segment_ranges: Vec<TimestampRange>,
    /// Codec and resolution of the destination file as actually written.
    written_video: WrittenVideo,
}

impl CameraVideoWriteResult {
    /// Creates the complete result of writing one camera video.
    #[must_use]
    pub fn new(segment_ranges: Vec<TimestampRange>, written_video: WrittenVideo) -> Self {
        Self {
            segment_ranges,
            written_video,
        }
    }

    /// Destination ranges positionally aligned with the requested segments.
    #[must_use]
    pub fn segment_ranges(&self) -> &[TimestampRange] {
        &self.segment_ranges
    }

    /// Codec and dimensions observed in the completed output.
    #[must_use]
    pub const fn written_video(&self) -> &WrittenVideo {
        &self.written_video
    }
}

/// Dependency-inversion seam for video handling: this crate identifies the
/// source segments, while a caller-provided media implementation performs the
/// cutting and concatenation.
pub trait VideoSegmentWriter {
    /// Writes `segments` (in new-episode order) concatenated into `dest_mp4`
    /// and returns, for each input segment in the same order, the timestamp
    /// range it occupies in the destination file, together with the codec and
    /// resolution that were actually written.
    ///
    /// # Errors
    ///
    /// Implementations return any error from reading, transcoding, or writing
    /// video data; `export_subset` wraps it in [`LerobotError::VideoWrite`].
    fn write_camera_video(
        &self,
        camera: &str,
        segments: &[SourceSegment],
        dest_mp4: &Path,
    ) -> Result<CameraVideoWriteResult, Box<dyn std::error::Error + Send + Sync>>;
}

/// Optional format-extension behavior applied inside a subset export.
///
/// Implementations can write extension-owned sidecars and update their own
/// namespaced `info.json` fields without coupling the `LeRobot` writer to a
/// particular product or metadata schema.
pub trait DatasetExportExtension {
    /// Writes extension-owned sidecars for the selected episodes.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError`] when the extension cannot produce a valid
    /// sidecar in `destination_root`.
    fn write_exported_episode_sidecars(
        &self,
        _selected_episodes: &[&Episode],
        _destination_root: &Path,
    ) -> Result<(), LerobotError> {
        Ok(())
    }

    /// Rewrites extension-owned fields after the core feature declarations
    /// have been updated for the exported video layout.
    fn rewrite_exported_info_document(
        &self,
        _exported_document_object: &mut serde_json::Map<String, Value>,
    ) {
    }
}

/// Summary of a completed [`export_subset`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportReport {
    /// Number of episodes written to the destination dataset.
    pub episodes_written: u32,
    /// Number of per-frame rows written across all exported episodes.
    pub frames_written: u64,
}

/// Episode indices selected for an export. The constructor is the boundary
/// where an external list becomes a valid export request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportEpisodes(NonEmpty<u32>);

impl TryFrom<Vec<u32>> for ExportEpisodes {
    type Error = LerobotError;

    fn try_from(episode_indices: Vec<u32>) -> Result<Self, Self::Error> {
        let unique_episode_count = episode_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len();
        if unique_episode_count != episode_indices.len() {
            return Err(LerobotError::InconsistentMetadata {
                detail: "export episode selection contains a duplicate index".to_owned(),
            });
        }
        NonEmpty::from_vec(episode_indices).map_or_else(
            || {
                Err(LerobotError::InconsistentMetadata {
                    detail: "export requires at least one episode".to_owned(),
                })
            },
            |episode_indices| Ok(Self(episode_indices)),
        )
    }
}

impl ExportEpisodes {
    /// Iterates in the caller's requested export order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().copied()
    }
}

/// Proof that the files for one exact export request are available locally.
///
/// The proof records the dataset cache, ordered episode selection, and video
/// keys it prepared. Export entry points verify all three before claiming the
/// destination, preventing a proof for one request from authorizing another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedExportInputs {
    dataset_root: PathBuf,
    episode_indices: Vec<u32>,
    video_keys: BTreeSet<String>,
}

impl PreparedExportInputs {
    /// Prepares a local-source export without I/O; all of its files are already
    /// local by definition.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Source`] for a remote source — those
    /// must go through [`Dataset::prepare_export_inputs`].
    pub fn for_local_dataset(
        dataset: &Dataset,
        episode_indices: &ExportEpisodes,
        video_keys: &[String],
    ) -> Result<Self, LerobotError> {
        if dataset.is_remote_cached() {
            return Err(LerobotError::Source {
                context: "preparing export inputs".to_owned(),
                detail: "remote dataset sources must prefetch export inputs \
                         (await Dataset::prepare_export_inputs)"
                    .to_owned(),
            });
        }
        Ok(Self::for_request(dataset, episode_indices, video_keys))
    }

    fn for_request(
        dataset: &Dataset,
        episode_indices: &ExportEpisodes,
        video_keys: &[String],
    ) -> Self {
        Self {
            dataset_root: dataset.root().to_path_buf(),
            episode_indices: episode_indices.iter().collect(),
            video_keys: video_keys.iter().cloned().collect(),
        }
    }

    fn verify_for_request(
        &self,
        dataset: &Dataset,
        episode_indices: &ExportEpisodes,
        video_keys: &[String],
    ) -> Result<(), LerobotError> {
        let requested_episode_indices: Vec<u32> = episode_indices.iter().collect();
        let requested_video_keys: BTreeSet<&str> = video_keys.iter().map(String::as_str).collect();
        let prepared_video_keys: BTreeSet<&str> =
            self.video_keys.iter().map(String::as_str).collect();
        if self.dataset_root != dataset.root()
            || self.episode_indices != requested_episode_indices
            || prepared_video_keys != requested_video_keys
        {
            return Err(LerobotError::Source {
                context: "validating prepared export inputs".to_owned(),
                detail: "prepared inputs do not match the dataset, episode order, and video keys requested by this export"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

impl Dataset {
    /// Materializes everything an export of `episode_indices` will read — the
    /// packed data parquets plus the packed video files of `video_keys` — and
    /// returns the proof token the export entry points require. A local
    /// dataset resolves immediately.
    ///
    /// Pass [`Dataset::video_keys`] for a re-encoding [`export_subset`] and
    /// the retained keys for [`export_subset_preserving_packed_videos`].
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Source`] / [`LerobotError::Io`] when a download
    /// fails and [`LerobotError::InconsistentMetadata`] for an invalid path
    /// template.
    pub async fn prepare_export_inputs(
        &self,
        episode_indices: &ExportEpisodes,
        video_keys: &[String],
    ) -> Result<PreparedExportInputs, LerobotError> {
        let episode_index_list: Vec<u32> = episode_indices.iter().collect();
        self.prefetch_episode_data(&episode_index_list).await?;
        self.prefetch_video_segments(&episode_index_list, video_keys)
            .await?;
        Ok(PreparedExportInputs::for_request(
            self,
            episode_indices,
            video_keys,
        ))
    }
}

/// Exports the given episodes of `src` (re-indexed `0..n` in the given order)
/// as a new self-contained `LeRobot` v3 dataset under `dest_root`.
///
/// The export mirrors the source parquet schemas field-for-field (including
/// the pandas `__index_level_0__` task column quirk) so that python `lerobot`
/// can load the result, and [`Dataset::open`] on `dest_root` round-trips.
///
/// Because every export re-encodes videos, the exported `meta/info.json`
/// video features are patched to the codec, resolution, and shape the
/// [`VideoSegmentWriter`] reports. Only the re-encodable RGB surface
/// ([`Dataset::video_keys`]) is written; other video modalities (depth,
/// unclassified) are dropped from the export together with their feature
/// declarations — use [`export_subset_preserving_packed_videos`] to retain
/// them losslessly. A canonical source's episode-annotation sidecar is
/// rewritten for the selection. The exported data parquet mirrors the
/// source's pandas schema metadata verbatim; that metadata only describes the
/// parquet's own (non-video) columns, so `meta/info.json` is the sole
/// authority for feature declarations such as video codec and resolution.
/// Per-episode `stats/*` values are copied verbatim, except for the rewritten
/// bookkeeping columns (`episode_index`, `index`, `task_index`) whose
/// min/max/mean/std/count are recomputed from the rewritten values.
///
/// The destination must not exist yet, be empty, or contain only
/// [`EXPORT_STAGING_OWNERSHIP_MANIFEST_FILE_NAME`], and may not overlap the
/// source dataset. While the export runs, `dest_root` holds a
/// `.lerobot-dataset-export-in-progress` marker file (removed on success and
/// failure),
/// so a concurrent export to the same destination fails cleanly.
///
/// # Errors
///
/// Returns [`LerobotError::ExportDestination`] when the destination is the
/// source, overlaps it, is non-empty, or is claimed by another export;
/// [`LerobotError::EpisodeNotFound`] for unknown episode indices;
/// [`LerobotError::VideoWrite`] when the [`VideoSegmentWriter`] fails; and
/// I/O, parquet, or consistency errors from reading the source or writing the
/// destination.
pub fn export_subset(
    src: &Dataset,
    episode_indices: &ExportEpisodes,
    dest_root: &Path,
    video_writer: &dyn VideoSegmentWriter,
    prepared_inputs: PreparedExportInputs,
) -> Result<ExportReport, LerobotError> {
    export_subset_with_extension(
        src,
        episode_indices,
        dest_root,
        video_writer,
        prepared_inputs,
        None,
    )
}

/// Variant of [`export_subset`] that applies an optional metadata extension.
///
/// # Errors
///
/// Returns the same errors as [`export_subset`], plus any error returned by
/// the extension while writing its sidecars.
pub fn export_subset_with_extension(
    src: &Dataset,
    episode_indices: &ExportEpisodes,
    dest_root: &Path,
    video_writer: &dyn VideoSegmentWriter,
    prepared_inputs: PreparedExportInputs,
    export_extension: Option<&dyn DatasetExportExtension>,
) -> Result<ExportReport, LerobotError> {
    prepared_inputs.verify_for_request(src, episode_indices, src.video_keys())?;
    let selected_episodes: Vec<&Episode> = episode_indices
        .iter()
        .map(|episode_index| {
            src.episode(episode_index)
                .ok_or(LerobotError::EpisodeNotFound {
                    index: episode_index,
                })
        })
        .collect::<Result<_, _>>()?;

    // Claim the destination BEFORE writing anything; the claim's marker file
    // is removed when `_destination_claim` drops, on success and on every
    // error path below.
    let _destination_claim = claim_export_destination(src.root(), dest_root)?;

    create_dir_all_with_context(&dest_root.join("meta"))?;

    let camera_write_results = write_videos(src, &selected_episodes, dest_root, video_writer)?;

    let source_task_table = read_source_task_table(src.root())?;
    let mut task_remap = TaskRemap::new(source_task_table.task_strings_by_index);
    let mut task_rewrite = TaskIndexRewrite::Compact(&mut task_remap);
    let (frames_written, bookkeeping_stats) = write_data_parquet(
        src,
        &selected_episodes,
        dest_root,
        ExportRangePlacement::WHOLE,
        &mut task_rewrite,
    )?;

    write_tasks_parquet(
        dest_root,
        &source_task_table.schema,
        &source_task_table.task_string_column_name,
        &task_remap.new_task_strings,
    )?;
    let exported_video_layout = ExportedVideoLayout::Reencoded(&camera_write_results);
    write_episodes_parquet(
        src,
        &selected_episodes,
        dest_root,
        ExportRangePlacement::WHOLE,
        exported_video_layout,
        &bookkeeping_stats,
    )?;
    if let Some(export_extension) = export_extension {
        export_extension.write_exported_episode_sidecars(&selected_episodes, dest_root)?;
    }
    write_export_info_json(
        src,
        selected_episodes.len(),
        frames_written,
        task_remap.new_task_strings.len(),
        exported_video_layout,
        dest_root,
        export_extension,
    )?;
    copy_stats_json_if_present(src.root(), dest_root)?;

    let episodes_written =
        u32::try_from(selected_episodes.len()).map_err(|_| LerobotError::InconsistentMetadata {
            detail: "export episode count exceeds u32 range".to_owned(),
        })?;
    Ok(ExportReport {
        episodes_written,
        frames_written,
    })
}

/// Exports selected episodes while copying their packed source video files
/// byte-for-byte instead of decoding or re-encoding them.
///
/// Data rows, task indices, episode indices, and episode metadata are still
/// rewritten into a self-contained, contiguous `LeRobot` v3 subset. A packed
/// video file can contain unselected episode segments; retaining those unused
/// bytes is the intentional tradeoff that makes this path fast, lossless, and
/// resumable at the file level. Only `retained_video_keys` are declared in the
/// destination and copied. A canonical source's episode-annotation sidecar is
/// rewritten for the selection. Aggregate `meta/stats.json` is deliberately
/// omitted because source-global statistics would be incorrect for the subset.
///
/// The destination rules are identical to [`export_subset`]: it must not
/// overlap the source and must be absent, empty, or coordinator-owned staging.
///
/// # Errors
///
/// Returns [`LerobotError::InconsistentMetadata`] for an empty selection,
/// unknown or duplicate retained video keys, missing episode camera segments,
/// or inconsistent source metadata; plus the same destination, parquet, and
/// I/O failures as [`export_subset`].
pub fn export_subset_preserving_packed_videos(
    src: &Dataset,
    episode_indices: &ExportEpisodes,
    retained_video_keys: &[String],
    dest_root: &Path,
    prepared_inputs: PreparedExportInputs,
) -> Result<ExportReport, LerobotError> {
    export_subset_preserving_packed_videos_with_extension(
        src,
        episode_indices,
        retained_video_keys,
        dest_root,
        prepared_inputs,
        None,
    )
}

/// Variant of [`export_subset_preserving_packed_videos`] that applies an
/// optional metadata extension.
///
/// # Errors
///
/// Returns the same errors as [`export_subset_preserving_packed_videos`], plus
/// any error returned by the extension while writing its sidecars.
pub fn export_subset_preserving_packed_videos_with_extension(
    src: &Dataset,
    episode_indices: &ExportEpisodes,
    retained_video_keys: &[String],
    dest_root: &Path,
    prepared_inputs: PreparedExportInputs,
    export_extension: Option<&dyn DatasetExportExtension>,
) -> Result<ExportReport, LerobotError> {
    let selected_episodes: Vec<&Episode> = episode_indices
        .iter()
        .map(|episode_index| {
            src.episode(episode_index)
                .ok_or(LerobotError::EpisodeNotFound {
                    index: episode_index,
                })
        })
        .collect::<Result<_, _>>()?;
    prepared_inputs.verify_for_request(src, episode_indices, retained_video_keys)?;
    let retained_video_keys = RetainedVideoKeys::parse(src, retained_video_keys)?;

    let _destination_claim = claim_export_destination(src.root(), dest_root)?;
    create_dir_all_with_context(&dest_root.join("meta"))?;
    copy_packed_video_files(src, &selected_episodes, &retained_video_keys, dest_root)?;

    let source_task_table = read_source_task_table(src.root())?;
    let mut task_remap = TaskRemap::new(source_task_table.task_strings_by_index);
    let mut task_rewrite = TaskIndexRewrite::Compact(&mut task_remap);
    let (frames_written, bookkeeping_stats) = write_data_parquet(
        src,
        &selected_episodes,
        dest_root,
        ExportRangePlacement::WHOLE,
        &mut task_rewrite,
    )?;
    write_tasks_parquet(
        dest_root,
        &source_task_table.schema,
        &source_task_table.task_string_column_name,
        &task_remap.new_task_strings,
    )?;
    let exported_video_layout = ExportedVideoLayout::PreservedPacked(&retained_video_keys);
    write_episodes_parquet(
        src,
        &selected_episodes,
        dest_root,
        ExportRangePlacement::WHOLE,
        exported_video_layout,
        &bookkeeping_stats,
    )?;
    if let Some(export_extension) = export_extension {
        export_extension.write_exported_episode_sidecars(&selected_episodes, dest_root)?;
    }
    write_export_info_json(
        src,
        selected_episodes.len(),
        frames_written,
        task_remap.new_task_strings.len(),
        exported_video_layout,
        dest_root,
        export_extension,
    )?;

    let episodes_written =
        u32::try_from(selected_episodes.len()).map_err(|_| LerobotError::InconsistentMetadata {
            detail: "export episode count exceeds u32 range".to_owned(),
        })?;
    Ok(ExportReport {
        episodes_written,
        frames_written,
    })
}

/// Exports ONE episode range of a larger canonical dataset into `dest_root`
/// as final-shaped artifacts: its data parquet at
/// `data/chunk-000/file-{output_file_index:03}.parquet`, its episodes
/// metadata file, and its packed video files at their source-relative
/// (final) keys. Frame rows keep SOURCE `task_index` values, and every
/// global (canonical episode index, global `index`, `dataset_from/to`) is
/// derived from `placement`, so concurrent ranges of one dataset are fully
/// independent. Deliberately absent: `meta/info.json` and
/// `meta/tasks.parquet` — the caller finalizes those once over all ranges.
///
/// # Errors
///
/// Same failure surface as [`export_subset_preserving_packed_videos`].
pub fn export_episode_range_preserving_packed_videos(
    src: &Dataset,
    episode_indices: &ExportEpisodes,
    retained_video_keys: &[String],
    dest_root: &Path,
    prepared_inputs: PreparedExportInputs,
    placement: ExportRangePlacement,
) -> Result<ExportReport, LerobotError> {
    let selected_episodes: Vec<&Episode> = episode_indices
        .iter()
        .map(|episode_index| {
            src.episode(episode_index)
                .ok_or(LerobotError::EpisodeNotFound {
                    index: episode_index,
                })
        })
        .collect::<Result<_, _>>()?;
    prepared_inputs.verify_for_request(src, episode_indices, retained_video_keys)?;
    let retained_video_keys = RetainedVideoKeys::parse(src, retained_video_keys)?;

    let _destination_claim = claim_export_destination(src.root(), dest_root)?;
    create_dir_all_with_context(&dest_root.join("meta"))?;
    copy_packed_video_files(src, &selected_episodes, &retained_video_keys, dest_root)?;

    let mut task_rewrite = TaskIndexRewrite::PreserveSource;
    let (frames_written, bookkeeping_stats) = write_data_parquet(
        src,
        &selected_episodes,
        dest_root,
        placement,
        &mut task_rewrite,
    )?;
    let exported_video_layout = ExportedVideoLayout::PreservedPacked(&retained_video_keys);
    write_episodes_parquet(
        src,
        &selected_episodes,
        dest_root,
        placement,
        exported_video_layout,
        &bookkeeping_stats,
    )?;

    let episodes_written =
        u32::try_from(selected_episodes.len()).map_err(|_| LerobotError::InconsistentMetadata {
            detail: "export episode count exceeds u32 range".to_owned(),
        })?;
    Ok(ExportReport {
        episodes_written,
        frames_written,
    })
}

/// Where an exported episode range sits inside a larger canonical dataset;
/// the whole-dataset export is the single-range case.
#[derive(Debug, Clone, Copy)]
pub struct ExportRangePlacement {
    /// Output data/episodes-metadata file index within chunk 0.
    output_file_index: u32,
    /// Canonical index of the range's first episode.
    episode_index_base: u32,
    /// Frames before this range: its first global `index`, which is also
    /// its first `dataset_from_index`.
    frames_before: u64,
}

impl ExportRangePlacement {
    /// The whole dataset as one range.
    pub const WHOLE: Self = Self {
        output_file_index: 0,
        episode_index_base: 0,
        frames_before: 0,
    };

    /// Describes where one independently exported range belongs in the final
    /// canonical dataset.
    #[must_use]
    pub const fn new(output_file_index: u32, episode_index_base: u32, frames_before: u64) -> Self {
        Self {
            output_file_index,
            episode_index_base,
            frames_before,
        }
    }

    /// Output data and episode-metadata file index within chunk zero.
    #[must_use]
    pub const fn output_file_index(self) -> u32 {
        self.output_file_index
    }

    /// Canonical index assigned to the range's first episode.
    #[must_use]
    pub const fn episode_index_base(self) -> u32 {
        self.episode_index_base
    }

    /// Number of canonical frames preceding this range.
    #[must_use]
    pub const fn frames_before(self) -> u64 {
        self.frames_before
    }
}

/// How exported frame rows' `task_index` values are written.
enum TaskIndexRewrite<'a> {
    /// Compact to first-appearance order; the whole-dataset export writes
    /// the compacted task table alongside.
    Compact(&'a mut TaskRemap),
    /// Keep SOURCE task indices verbatim, so concurrent range exports stay
    /// independent; the caller ships the source task table unchanged.
    PreserveSource,
}

impl TaskIndexRewrite<'_> {
    fn rewrite(&mut self, source_task_index: i64) -> Result<i64, LerobotError> {
        match self {
            Self::Compact(task_remap) => task_remap.remap(source_task_index),
            Self::PreserveSource => Ok(source_task_index),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ExportedVideoLayout<'a> {
    Reencoded(&'a BTreeMap<String, CameraVideoWriteResult>),
    PreservedPacked(&'a RetainedVideoKeys),
}

/// Parsed proof that a packed export has a non-empty, unique subset of source
/// video keys. Downstream copy/metadata functions consume this type and do not
/// repeat boundary validation.
#[derive(Debug, Clone)]
struct RetainedVideoKeys(BTreeSet<String>);

impl RetainedVideoKeys {
    fn parse(src: &Dataset, requested_video_keys: &[String]) -> Result<Self, LerobotError> {
        if requested_video_keys.is_empty() {
            return Err(LerobotError::InconsistentMetadata {
                detail: "packed-video export requires at least one retained video key".to_owned(),
            });
        }
        let parsed_video_keys: BTreeSet<String> = requested_video_keys.iter().cloned().collect();
        if parsed_video_keys.len() != requested_video_keys.len() {
            return Err(LerobotError::InconsistentMetadata {
                detail: "retained video keys must be unique".to_owned(),
            });
        }
        // Packed exports copy encoded video whole, so any CLASSIFIED view —
        // RGB or depth — may be retained; only unclassified keys are refused.
        let classified_video_keys = src.all_video_keys();
        for parsed_video_key in &parsed_video_keys {
            if !classified_video_keys.contains(parsed_video_key) {
                return Err(LerobotError::InconsistentMetadata {
                    detail: format!(
                        "retained video key {parsed_video_key:?} is not available in the source dataset"
                    ),
                });
            }
        }
        Ok(Self(parsed_video_keys))
    }

    fn as_set(&self) -> &BTreeSet<String> {
        &self.0
    }
}

/// Versioned ownership proof that a higher-level export coordinator may place
/// in a staging directory before calling [`export_subset`]. The coordinator is
/// responsible for validating its contents; this crate only reserves the file
/// name so its destination claim does not mistake that proof for dataset data.
pub const EXPORT_STAGING_OWNERSHIP_MANIFEST_FILE_NAME: &str =
    ".lerobot-dataset-export-staging-owner.json";

/// Marker file created with `create_new` while an export writes into a
/// destination; its presence means "an export currently owns this directory".
const EXPORT_CLAIM_MARKER_FILE_NAME: &str = ".lerobot-dataset-export-in-progress";

/// RAII claim on an export destination directory: dropping it removes the
/// marker file, on the success path and on every failure path alike.
struct ExportDestinationClaim {
    marker_path: PathBuf,
}

impl Drop for ExportDestinationClaim {
    fn drop(&mut self) {
        if let Err(remove_error) = fs::remove_file(&self.marker_path) {
            tracing::warn!(
                marker = %self.marker_path.display(),
                error = %remove_error,
                "failed to remove export destination claim marker"
            );
        }
    }
}

/// Validates `dest_root` against `source_root` and atomically claims it.
///
/// The destination must not overlap the source dataset in either direction and
/// must not exist yet, be an empty directory, or contain only a coordinator
/// staging ownership manifest. Claiming creates the
/// [`EXPORT_CLAIM_MARKER_FILE_NAME`] marker with `create_new`, so of two
/// concurrent exports racing for the same destination exactly one wins and the
/// other fails with [`ExportDestinationProblem::AlreadyClaimed`].
fn claim_export_destination(
    source_root: &Path,
    dest_root: &Path,
) -> Result<ExportDestinationClaim, LerobotError> {
    // `Dataset::open` already canonicalizes its root, but claim callers in
    // tests may not; re-canonicalizing an already-canonical path is cheap.
    let canonical_source_root =
        fs::canonicalize(source_root).map_err(|source| LerobotError::Io {
            path: source_root.to_path_buf(),
            source,
        })?;
    let resolved_dest_root = resolve_destination_root(dest_root)?;

    let destination_problem = |problem: ExportDestinationProblem| LerobotError::ExportDestination {
        dest: dest_root.to_path_buf(),
        problem,
    };

    if resolved_dest_root == canonical_source_root {
        return Err(destination_problem(ExportDestinationProblem::SameAsSource));
    }
    if resolved_dest_root.starts_with(&canonical_source_root) {
        return Err(destination_problem(ExportDestinationProblem::InsideSource));
    }
    if canonical_source_root.starts_with(&resolved_dest_root) {
        return Err(destination_problem(
            ExportDestinationProblem::ContainsSource,
        ));
    }

    match fs::metadata(dest_root) {
        Ok(dest_metadata) if !dest_metadata.is_dir() => {
            return Err(destination_problem(ExportDestinationProblem::NotADirectory));
        }
        Ok(_) => {
            let directory_entries = fs::read_dir(dest_root).map_err(|source| LerobotError::Io {
                path: dest_root.to_path_buf(),
                source,
            })?;
            for entry_result in directory_entries {
                let entry = entry_result.map_err(|source| LerobotError::Io {
                    path: dest_root.to_path_buf(),
                    source,
                })?;
                // Claim and coordinator-owned staging markers are not dataset
                // content. The claim marker is resolved by `create_new` below;
                // the ownership manifest remains in the completed export as
                // proof that recursive staging cleanup is safe.
                if entry.file_name() != EXPORT_CLAIM_MARKER_FILE_NAME
                    && entry.file_name() != EXPORT_STAGING_OWNERSHIP_MANIFEST_FILE_NAME
                {
                    return Err(destination_problem(ExportDestinationProblem::NotEmpty));
                }
            }
        }
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            create_dir_all_with_context(dest_root)?;
        }
        Err(metadata_error) => {
            return Err(LerobotError::Io {
                path: dest_root.to_path_buf(),
                source: metadata_error,
            });
        }
    }

    let marker_path = dest_root.join(EXPORT_CLAIM_MARKER_FILE_NAME);
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker_path)
    {
        Ok(_marker_file) => Ok(ExportDestinationClaim { marker_path }),
        Err(open_error) if open_error.kind() == std::io::ErrorKind::AlreadyExists => Err(
            destination_problem(ExportDestinationProblem::AlreadyClaimed),
        ),
        Err(open_error) => Err(LerobotError::Io {
            path: marker_path,
            source: open_error,
        }),
    }
}

/// Resolves `dest_root` (which may not exist yet) to an absolute, symlink-free
/// path for overlap comparisons: the deepest EXISTING ancestor is
/// canonicalized and the not-yet-created remainder is appended after lexical
/// `.`/`..` folding.
fn resolve_destination_root(dest_root: &Path) -> Result<PathBuf, LerobotError> {
    let absolute_dest = if dest_root.is_absolute() {
        dest_root.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| LerobotError::Io {
                path: dest_root.to_path_buf(),
                source,
            })?
            .join(dest_root)
    };

    let deepest_existing_ancestor = absolute_dest
        .ancestors()
        .find(|ancestor| ancestor.exists())
        .expect("an absolute path has an existing root ancestor");
    let canonical_ancestor =
        fs::canonicalize(deepest_existing_ancestor).map_err(|source| LerobotError::Io {
            path: deepest_existing_ancestor.to_path_buf(),
            source,
        })?;

    let unresolved_remainder = absolute_dest
        .strip_prefix(deepest_existing_ancestor)
        .expect("Path::ancestors yields prefixes of the path");
    let mut resolved = canonical_ancestor;
    for component in unresolved_remainder.components() {
        match component {
            Component::Normal(component_name) => resolved.push(component_name),
            Component::ParentDir => {
                resolved.pop();
            }
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    Ok(resolved)
}

/// Calls the video writer once per camera (in `video_keys` order) with that
/// camera's segments in new-episode order.
fn write_videos(
    src: &Dataset,
    selected_episodes: &[&Episode],
    dest_root: &Path,
    video_writer: &dyn VideoSegmentWriter,
) -> Result<BTreeMap<String, CameraVideoWriteResult>, LerobotError> {
    let mut camera_write_results = BTreeMap::new();
    for camera in src.video_keys() {
        let segments: Vec<SourceSegment> = selected_episodes
            .iter()
            .map(|episode| {
                let segment = episode.video_segments().get(camera).ok_or_else(|| {
                    LerobotError::InconsistentMetadata {
                        detail: format!(
                            "episode {} has no video segment for camera {camera:?}",
                            episode.episode_index()
                        ),
                    }
                })?;
                Ok(SourceSegment::new(
                    // Already materialized by export-input preparation; a
                    // local source returns the segment's source path.
                    src.local_segment_path(segment)?,
                    segment.timestamp_range(),
                ))
            })
            .collect::<Result<_, LerobotError>>()?;

        let relative_dest =
            render_path_template(src.info().video_path_template(), Some(camera), 0, 0)?;
        let dest_mp4 = relative_dest.join_under(dest_root);
        if let Some(parent) = dest_mp4.parent() {
            create_dir_all_with_context(parent)?;
        }

        let write_result = video_writer
            .write_camera_video(camera, &segments, &dest_mp4)
            .map_err(|source| LerobotError::VideoWrite {
                camera: camera.clone(),
                source,
            })?;
        if write_result.segment_ranges().len() != selected_episodes.len() {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "video writer returned {} ranges for camera {camera:?} but {} episodes were requested",
                    write_result.segment_ranges().len(),
                    selected_episodes.len()
                ),
            });
        }
        camera_write_results.insert(camera.clone(), write_result);
    }
    Ok(camera_write_results)
}

fn copy_packed_video_files(
    src: &Dataset,
    selected_episodes: &[&Episode],
    retained_video_keys: &RetainedVideoKeys,
    dest_root: &Path,
) -> Result<(), LerobotError> {
    let mut copied_relative_paths = BTreeSet::new();
    for source_episode in selected_episodes {
        for retained_video_key in retained_video_keys.as_set() {
            let source_segment = source_episode
                .video_segments()
                .get(retained_video_key)
                .ok_or_else(|| LerobotError::InconsistentMetadata {
                    detail: format!(
                        "episode {} has no video segment for retained key {retained_video_key:?}",
                        source_episode.episode_index()
                    ),
                })?;
            let relative_video_path = src.video_segment_relative_key(source_segment)?;
            if !copied_relative_paths.insert(relative_video_path.clone()) {
                continue;
            }
            let source_video_path = src.local_segment_path(source_segment)?;
            let destination_video_path = dest_root.join(&relative_video_path);
            if let Some(destination_parent) = destination_video_path.parent() {
                create_dir_all_with_context(destination_parent)?;
            }
            fs::copy(&source_video_path, &destination_video_path).map_err(|source| {
                LerobotError::Io {
                    path: destination_video_path,
                    source,
                }
            })?;
        }
    }
    Ok(())
}

struct SourceTaskTable {
    schema: SchemaRef,
    task_string_column_name: String,
    task_strings_by_index: HashMap<i64, String>,
}

fn read_source_task_table(source_root: &Path) -> Result<SourceTaskTable, LerobotError> {
    let tasks_path = source_root.join("meta").join("tasks.parquet");
    let builder = open_parquet_reader_builder(&tasks_path)?;
    let schema = builder.schema().clone();

    // LeRobot serializes the task text as the pandas DataFrame *index*, which
    // parquet stores under the literal column name "__index_level_0__".
    let task_string_column_name = ["__index_level_0__", "task"]
        .into_iter()
        .find(|candidate| schema.column_with_name(candidate).is_some())
        .ok_or_else(|| LerobotError::MissingColumn {
            name: "__index_level_0__".to_owned(),
        })?
        .to_owned();

    let reader = builder.build().map_err(|source| LerobotError::Parquet {
        path: tasks_path.clone(),
        source,
    })?;
    let mut task_strings_by_index = HashMap::new();
    for batch_result in reader {
        let batch = batch_result?;
        let task_index_column = require_int64_column(&batch, "task_index")?;
        let task_string_column = require_string_column(&batch, &task_string_column_name)?;
        for row in 0..batch.num_rows() {
            let task_index = int64_row_value(task_index_column, row, "task_index")?;
            if task_string_column.is_null(row) {
                return Err(LerobotError::InconsistentMetadata {
                    detail: format!("null task string for task_index {task_index}"),
                });
            }
            task_strings_by_index.insert(task_index, task_string_column.value(row).to_owned());
        }
    }
    Ok(SourceTaskTable {
        schema,
        task_string_column_name,
        task_strings_by_index,
    })
}

/// Remaps source `task_index` values to a compact new task table, assigning
/// new indices in order of first appearance in the exported frames.
struct TaskRemap {
    source_task_strings_by_index: HashMap<i64, String>,
    new_index_by_source_index: HashMap<i64, i64>,
    new_task_strings: Vec<String>,
}

impl TaskRemap {
    fn new(source_task_strings_by_index: HashMap<i64, String>) -> Self {
        Self {
            source_task_strings_by_index,
            new_index_by_source_index: HashMap::new(),
            new_task_strings: Vec::new(),
        }
    }

    fn remap(&mut self, source_task_index: i64) -> Result<i64, LerobotError> {
        if let Some(&new_index) = self.new_index_by_source_index.get(&source_task_index) {
            return Ok(new_index);
        }
        let task_string = self
            .source_task_strings_by_index
            .get(&source_task_index)
            .ok_or_else(|| LerobotError::InconsistentMetadata {
                detail: format!(
                    "frame references task_index {source_task_index} absent from meta/tasks.parquet"
                ),
            })?;
        let new_index = i64_from_usize(self.new_task_strings.len());
        self.new_task_strings.push(task_string.clone());
        self.new_index_by_source_index
            .insert(source_task_index, new_index);
        Ok(new_index)
    }
}

/// Population statistics accumulated over one rewritten int64 bookkeeping
/// column of one exported episode.
#[derive(Debug, Clone, Copy, Default)]
struct RewrittenColumnStats {
    count: u64,
    minimum: i64,
    maximum: i64,
    /// First recorded value; sums accumulate deltas from it so `mean` and
    /// `population_std` stay numerically stable for large global indices.
    baseline: i64,
    sum_of_deltas: f64,
    sum_of_squared_deltas: f64,
}

impl RewrittenColumnStats {
    fn record(&mut self, value: i64) {
        if self.count == 0 {
            self.baseline = value;
            self.minimum = value;
            self.maximum = value;
        } else {
            self.minimum = self.minimum.min(value);
            self.maximum = self.maximum.max(value);
        }
        // Within one episode the bookkeeping values span at most the episode's
        // frame count, so the delta always fits an f64 mantissa.
        #[allow(clippy::cast_precision_loss)]
        let delta = (value - self.baseline) as f64;
        self.sum_of_deltas += delta;
        self.sum_of_squared_deltas += delta * delta;
        self.count += 1;
    }

    /// Mean of the recorded values; only meaningful when `count > 0`.
    fn mean(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let count = self.count as f64;
        #[allow(clippy::cast_precision_loss)]
        let baseline = self.baseline as f64;
        baseline + self.sum_of_deltas / count
    }

    /// Population (ddof = 0) standard deviation, matching what python
    /// `lerobot` writes; only meaningful when `count > 0`.
    fn population_std(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let count = self.count as f64;
        let mean_delta = self.sum_of_deltas / count;
        let variance = self.sum_of_squared_deltas / count - mean_delta * mean_delta;
        variance.max(0.0).sqrt()
    }
}

/// Recomputed per-episode statistics for the three bookkeeping columns the
/// export rewrites; every other `stats/*` value is copied verbatim.
#[derive(Debug, Clone, Copy, Default)]
struct EpisodeBookkeepingStats {
    episode_index: RewrittenColumnStats,
    index: RewrittenColumnStats,
    task_index: RewrittenColumnStats,
}

/// One decoded source data shard retained while consecutive selected episodes
/// reference it. Prepared v3 datasets pack many episodes per parquet file; a
/// bounded one-file cache avoids reopening and rescanning that shard once per
/// episode without allowing memory to grow with the full dataset.
struct CachedDataParquet {
    source_path: PathBuf,
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

impl CachedDataParquet {
    fn read(source_path: PathBuf) -> Result<Self, LerobotError> {
        let builder = open_parquet_reader_builder(&source_path)?;
        let schema = builder.schema().clone();
        let reader = builder.build().map_err(|source| LerobotError::Parquet {
            path: source_path.clone(),
            source,
        })?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source_path,
            schema,
            batches,
        })
    }
}

fn write_data_parquet(
    src: &Dataset,
    selected_episodes: &[&Episode],
    dest_root: &Path,
    placement: ExportRangePlacement,
    task_rewrite: &mut TaskIndexRewrite<'_>,
) -> Result<(u64, Vec<EpisodeBookkeepingStats>), LerobotError> {
    let data_path_template = src.info().data_path_template();
    let dest_path = render_path_template(data_path_template, None, 0, placement.output_file_index)?
        .join_under(dest_root);
    if let Some(parent) = dest_path.parent() {
        create_dir_all_with_context(parent)?;
    }

    let mut arrow_writer: Option<(ArrowWriter<File>, SchemaRef)> = None;
    let mut next_global_index: i64 =
        i64::try_from(placement.frames_before).map_err(|_| LerobotError::InconsistentMetadata {
            detail: "range placement frames_before exceeds i64 range".to_owned(),
        })?;
    let mut total_frames_written: u64 = 0;
    let mut per_episode_bookkeeping_stats: Vec<EpisodeBookkeepingStats> =
        Vec::with_capacity(selected_episodes.len());
    let mut cached_source_data: Option<CachedDataParquet> = None;

    for (new_episode_position, source_episode) in selected_episodes.iter().enumerate() {
        // Resolve through the download seam so an object-store source fetches
        // the data parquet into the local cache; identical to the manual
        // template render for a local source.
        let source_path = src.local_episode_data_parquet_path(source_episode.episode_index())?;
        if cached_source_data
            .as_ref()
            .is_none_or(|cached_data| cached_data.source_path != source_path)
        {
            cached_source_data = Some(CachedDataParquet::read(source_path.clone())?);
        }
        let cached_source_data = cached_source_data
            .as_ref()
            .expect("source data cache was populated for the selected episode");
        if arrow_writer.is_none() {
            // The destination mirrors the first source file's schema
            // field-for-field, including its pandas schema metadata.
            let source_schema = cached_source_data.schema.clone();
            arrow_writer = Some((
                create_arrow_writer(&dest_path, source_schema.clone())?,
                source_schema,
            ));
        }
        let (writer, writer_schema) = arrow_writer
            .as_mut()
            .expect("writer initialized on first episode");

        let target_episode_value = i64::from(source_episode.episode_index());
        let new_episode_index =
            i64::from(placement.episode_index_base) + i64_from_usize(new_episode_position);
        let mut rows_written_for_episode: u64 = 0;
        let mut episode_stats = EpisodeBookkeepingStats::default();

        for batch in &cached_source_data.batches {
            let episode_column = require_int64_column(batch, "episode_index")?;
            for (run_start, run_length) in
                contiguous_runs_of_value(episode_column, target_episode_value)
            {
                let sliced = batch.slice(run_start, run_length);
                let rewritten = rewrite_frame_columns(
                    &sliced,
                    writer_schema,
                    new_episode_index,
                    &mut next_global_index,
                    task_rewrite,
                    &mut episode_stats,
                )?;
                writer
                    .write(&rewritten)
                    .map_err(|source| LerobotError::Parquet {
                        path: dest_path.clone(),
                        source,
                    })?;
                rows_written_for_episode += u64_from_usize(run_length);
            }
        }

        if rows_written_for_episode != source_episode.length_frames() {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "episode {} has {} data rows but metadata declares length {}",
                    source_episode.episode_index(),
                    rows_written_for_episode,
                    source_episode.length_frames()
                ),
            });
        }
        total_frames_written += rows_written_for_episode;
        per_episode_bookkeeping_stats.push(episode_stats);
    }

    let (writer, _) = arrow_writer.expect("at least one episode was selected");
    writer.close().map_err(|source| LerobotError::Parquet {
        path: dest_path,
        source,
    })?;
    Ok((total_frames_written, per_episode_bookkeeping_stats))
}

/// Rewrites the bookkeeping columns of a run of frames: `episode_index`
/// becomes the new index, the global `index` column is recomputed to stay
/// contiguous from 0, and `task_index` is remapped into the new task table.
/// Per-episode `frame_index` and `timestamp` (and all other columns) are
/// copied verbatim. The rewritten values are also recorded into
/// `episode_stats` so the exported per-episode `stats/*` can be recomputed.
///
/// Output columns are resolved BY NAME from `sliced_batch`'s own schema, not
/// by position in `writer_schema`: different data files of one dataset may
/// order the same fields differently.
fn rewrite_frame_columns(
    sliced_batch: &RecordBatch,
    writer_schema: &SchemaRef,
    new_episode_index: i64,
    next_global_index: &mut i64,
    task_rewrite: &mut TaskIndexRewrite<'_>,
    episode_stats: &mut EpisodeBookkeepingStats,
) -> Result<RecordBatch, LerobotError> {
    let row_count = sliced_batch.num_rows();
    if sliced_batch.num_columns() != writer_schema.fields().len() {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!(
                "data file batch has {} columns but the first selected data file has {}",
                sliced_batch.num_columns(),
                writer_schema.fields().len()
            ),
        });
    }
    let mut columns: Vec<ArrayRef> = writer_schema
        .fields()
        .iter()
        .map(|field| {
            sliced_batch
                .column_by_name(field.name())
                .cloned()
                .ok_or_else(|| LerobotError::MissingColumn {
                    name: field.name().clone(),
                })
        })
        .collect::<Result<_, _>>()?;

    let episode_position = require_field_position(writer_schema, "episode_index")?;
    let index_position = require_field_position(writer_schema, "index")?;
    let task_position = require_field_position(writer_schema, "task_index")?;

    let source_task_column = columns[task_position]
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: "column \"task_index\" is not int64".to_owned(),
        })?;
    let mut remapped_task_values = Vec::with_capacity(row_count);
    for row in 0..row_count {
        let source_task_index = int64_row_value(source_task_column, row, "task_index")?;
        let remapped_task_index = task_rewrite.rewrite(source_task_index)?;
        remapped_task_values.push(remapped_task_index);
        episode_stats.task_index.record(remapped_task_index);
    }

    let global_index_start = *next_global_index;
    let global_index_end = global_index_start + i64_from_usize(row_count);
    *next_global_index = global_index_end;
    for global_index in global_index_start..global_index_end {
        episode_stats.index.record(global_index);
        episode_stats.episode_index.record(new_episode_index);
    }

    columns[episode_position] =
        Arc::new(Int64Array::from(vec![new_episode_index; row_count])) as ArrayRef;
    columns[index_position] = Arc::new(Int64Array::from_iter_values(
        global_index_start..global_index_end,
    )) as ArrayRef;
    columns[task_position] = Arc::new(Int64Array::from(remapped_task_values)) as ArrayRef;

    RecordBatch::try_new(writer_schema.clone(), columns).map_err(LerobotError::from)
}

fn contiguous_runs_of_value(column: &Int64Array, target_value: i64) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut current_run_start: Option<usize> = None;
    for row in 0..column.len() {
        let row_matches = !column.is_null(row) && column.value(row) == target_value;
        match (row_matches, current_run_start) {
            (true, None) => current_run_start = Some(row),
            (false, Some(run_start)) => {
                runs.push((run_start, row - run_start));
                current_run_start = None;
            }
            _ => {}
        }
    }
    if let Some(run_start) = current_run_start {
        runs.push((run_start, column.len() - run_start));
    }
    runs
}

fn write_tasks_parquet(
    dest_root: &Path,
    source_schema: &SchemaRef,
    task_string_column_name: &str,
    new_task_strings: &[String],
) -> Result<(), LerobotError> {
    let dest_path = dest_root.join("meta").join("tasks.parquet");

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(source_schema.fields().len());
    for field in source_schema.fields() {
        let column: ArrayRef = if field.name() == "task_index" {
            Arc::new(Int64Array::from_iter_values(
                (0..new_task_strings.len()).map(i64_from_usize),
            ))
        } else if field.name() == task_string_column_name {
            Arc::new(StringArray::from(new_task_strings.to_vec()))
        } else {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "cannot synthesize unexpected meta/tasks.parquet column {:?}",
                    field.name()
                ),
            });
        };
        columns.push(column);
    }
    let batch = RecordBatch::try_new(source_schema.clone(), columns)?;

    let mut writer = create_arrow_writer(&dest_path, source_schema.clone())?;
    writer
        .write(&batch)
        .map_err(|source| LerobotError::Parquet {
            path: dest_path.clone(),
            source,
        })?;
    writer.close().map_err(|source| LerobotError::Parquet {
        path: dest_path,
        source,
    })?;
    Ok(())
}

fn write_episodes_parquet(
    src: &Dataset,
    selected_episodes: &[&Episode],
    dest_root: &Path,
    placement: ExportRangePlacement,
    exported_video_layout: ExportedVideoLayout<'_>,
    bookkeeping_stats: &[EpisodeBookkeepingStats],
) -> Result<(), LerobotError> {
    let retained_video_keys = match exported_video_layout {
        ExportedVideoLayout::Reencoded(camera_write_results) => {
            camera_write_results.keys().cloned().collect()
        }
        ExportedVideoLayout::PreservedPacked(retained_video_keys) => {
            retained_video_keys.as_set().clone()
        }
    };
    let source_rows = read_source_episode_rows(src, &retained_video_keys)?;

    // The episodes-metadata path template is fixed in the v3 format (it is
    // not declared in info.json).
    let dest_path = dest_root
        .join("meta")
        .join("episodes")
        .join("chunk-000")
        .join(format!("file-{:03}.parquet", placement.output_file_index));
    if let Some(parent) = dest_path.parent() {
        create_dir_all_with_context(parent)?;
    }
    let mut writer = create_arrow_writer(&dest_path, source_rows.schema.clone())?;

    let mut next_dataset_from_index: i64 =
        i64::try_from(placement.frames_before).map_err(|_| LerobotError::InconsistentMetadata {
            detail: "range placement frames_before exceeds i64 range".to_owned(),
        })?;
    for (new_episode_position, source_episode) in selected_episodes.iter().enumerate() {
        let rewritten = rewrite_episode_metadata_row(
            &source_rows,
            source_episode,
            new_episode_position,
            placement,
            &mut next_dataset_from_index,
            exported_video_layout,
            &bookkeeping_stats[new_episode_position],
        )?;
        writer
            .write(&rewritten)
            .map_err(|source| LerobotError::Parquet {
                path: dest_path.clone(),
                source,
            })?;
    }

    writer.close().map_err(|source| LerobotError::Parquet {
        path: dest_path,
        source,
    })?;
    Ok(())
}

struct SourceEpisodeRows {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    row_location_by_episode_index: HashMap<i64, (usize, usize)>,
}

/// Re-reads the source episode metadata WITHOUT the stats/* projection so
/// each episode's stats values can be copied into the export verbatim.
fn read_source_episode_rows(
    src: &Dataset,
    retained_video_keys: &BTreeSet<String>,
) -> Result<SourceEpisodeRows, LerobotError> {
    let source_files = enumerate_chunked_parquet_files(&src.root().join("meta").join("episodes"))?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut schema: Option<SchemaRef> = None;
    for source_path in &source_files {
        let builder = open_parquet_reader_builder(source_path)?;
        let source_schema = builder.schema().clone();
        let retained_column_positions =
            retained_episode_metadata_column_positions(&source_schema, retained_video_keys);
        let projected_schema = Arc::new(Schema::new_with_metadata(
            retained_column_positions
                .iter()
                .map(|&position| source_schema.field(position).clone())
                .collect::<Vec<_>>(),
            source_schema.metadata().clone(),
        ));
        if let Some(expected_schema) = &schema {
            if expected_schema != &projected_schema {
                return Err(LerobotError::InconsistentMetadata {
                    detail: format!(
                        "episode metadata file {} has a different retained schema",
                        source_path.display()
                    ),
                });
            }
        } else {
            schema = Some(projected_schema.clone());
        }
        let reader = builder.build().map_err(|source| LerobotError::Parquet {
            path: source_path.clone(),
            source,
        })?;
        for batch_result in reader {
            let source_batch = batch_result?;
            let projected_columns = retained_column_positions
                .iter()
                .map(|&position| source_batch.column(position).clone())
                .collect();
            batches.push(RecordBatch::try_new(
                projected_schema.clone(),
                projected_columns,
            )?);
        }
    }
    let schema = schema.ok_or_else(|| LerobotError::InconsistentMetadata {
        detail: "source dataset has no episode metadata parquet files".to_owned(),
    })?;

    let mut row_location_by_episode_index: HashMap<i64, (usize, usize)> = HashMap::new();
    for (batch_position, batch) in batches.iter().enumerate() {
        let episode_column = require_int64_column(batch, "episode_index")?;
        for row in 0..batch.num_rows() {
            if !episode_column.is_null(row) {
                row_location_by_episode_index
                    .insert(episode_column.value(row), (batch_position, row));
            }
        }
    }

    Ok(SourceEpisodeRows {
        schema,
        batches,
        row_location_by_episode_index,
    })
}

fn retained_episode_metadata_column_positions(
    source_schema: &SchemaRef,
    retained_video_keys: &BTreeSet<String>,
) -> Vec<usize> {
    source_schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| {
            episode_metadata_video_key(field.name())
                .is_none_or(|video_key| retained_video_keys.contains(video_key))
        })
        .map(|(position, _)| position)
        .collect()
}

fn episode_metadata_video_key(column_name: &str) -> Option<&str> {
    column_name
        .strip_prefix("videos/")?
        .rsplit_once('/')
        .map(|(video_key, _)| video_key)
}

/// Copies one source episode metadata row (tasks, length, most `stats/*`) and
/// rewrites its canonical episode/data indices. A re-encoded layout also
/// writes new video chunk/file indices and timestamps; a packed layout retains
/// the source segment locations. Bookkeeping `stats/*` families follow the
/// values chosen by the layout.
fn rewrite_episode_metadata_row(
    source_rows: &SourceEpisodeRows,
    source_episode: &Episode,
    new_episode_position: usize,
    placement: ExportRangePlacement,
    next_dataset_from_index: &mut i64,
    exported_video_layout: ExportedVideoLayout<'_>,
    episode_stats: &EpisodeBookkeepingStats,
) -> Result<RecordBatch, LerobotError> {
    let schema = &source_rows.schema;
    let (batch_position, row) = *source_rows
        .row_location_by_episode_index
        .get(&i64::from(source_episode.episode_index()))
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: format!(
                "episode {} vanished from episode metadata during export",
                source_episode.episode_index()
            ),
        })?;
    let single_row = source_rows.batches[batch_position].slice(row, 1);
    let mut columns = single_row.columns().to_vec();

    let output_file_index = i64::from(placement.output_file_index);
    set_int64_value(
        schema,
        &mut columns,
        "episode_index",
        i64::from(placement.episode_index_base) + i64_from_usize(new_episode_position),
    )?;
    set_int64_value(schema, &mut columns, "data/chunk_index", 0)?;
    set_int64_value(schema, &mut columns, "data/file_index", output_file_index)?;
    set_int64_value_if_present(schema, &mut columns, "meta/episodes/chunk_index", 0);
    set_int64_value_if_present(
        schema,
        &mut columns,
        "meta/episodes/file_index",
        output_file_index,
    );

    let episode_length = i64::try_from(source_episode.length_frames()).map_err(|_| {
        LerobotError::InconsistentMetadata {
            detail: format!(
                "episode {} length {} exceeds i64 range",
                source_episode.episode_index(),
                source_episode.length_frames()
            ),
        }
    })?;
    let dataset_to_index = *next_dataset_from_index + episode_length;
    set_int64_value(
        schema,
        &mut columns,
        "dataset_from_index",
        *next_dataset_from_index,
    )?;
    set_int64_value(schema, &mut columns, "dataset_to_index", dataset_to_index)?;
    *next_dataset_from_index = dataset_to_index;

    match exported_video_layout {
        ExportedVideoLayout::Reencoded(camera_write_results) => {
            for (camera, write_result) in camera_write_results {
                let range = write_result.segment_ranges()[new_episode_position];
                rewrite_episode_video_location(schema, &mut columns, camera, 0, 0, range)?;
            }
        }
        ExportedVideoLayout::PreservedPacked(retained_video_keys) => {
            for camera in retained_video_keys.as_set() {
                let source_segment =
                    source_episode.video_segments().get(camera).ok_or_else(|| {
                        LerobotError::InconsistentMetadata {
                            detail: format!(
                                "episode {} has no packed segment for retained key {camera:?}",
                                source_episode.episode_index()
                            ),
                        }
                    })?;
                rewrite_episode_video_location(
                    schema,
                    &mut columns,
                    camera,
                    source_segment.chunk_index(),
                    source_segment.file_index(),
                    source_segment.timestamp_range(),
                )?;
            }
        }
    }

    overwrite_bookkeeping_stats(
        schema,
        &mut columns,
        "episode_index",
        &episode_stats.episode_index,
    )?;
    overwrite_bookkeeping_stats(schema, &mut columns, "index", &episode_stats.index)?;
    overwrite_bookkeeping_stats(
        schema,
        &mut columns,
        "task_index",
        &episode_stats.task_index,
    )?;

    RecordBatch::try_new(schema.clone(), columns).map_err(LerobotError::from)
}

fn rewrite_episode_video_location(
    schema: &SchemaRef,
    columns: &mut [ArrayRef],
    camera: &str,
    chunk_index: u32,
    file_index: u32,
    timestamp_range: TimestampRange,
) -> Result<(), LerobotError> {
    set_int64_value(
        schema,
        columns,
        &format!("videos/{camera}/chunk_index"),
        i64::from(chunk_index),
    )?;
    set_int64_value(
        schema,
        columns,
        &format!("videos/{camera}/file_index"),
        i64::from(file_index),
    )?;
    set_float64_value(
        schema,
        columns,
        &format!("videos/{camera}/from_timestamp"),
        timestamp_range.start_seconds(),
    )?;
    set_float64_value(
        schema,
        columns,
        &format!("videos/{camera}/to_timestamp"),
        timestamp_range.end_seconds(),
    )?;
    Ok(())
}

/// Overwrites the exported `stats/{column}/{min,max,mean,std,count}` cells for
/// one rewritten bookkeeping column with statistics recomputed from the
/// rewritten values, mirroring the source's single-element-list structure.
/// Datasets without per-episode stats columns are left untouched.
fn overwrite_bookkeeping_stats(
    schema: &SchemaRef,
    columns: &mut [ArrayRef],
    stats_column_name: &str,
    stats: &RewrittenColumnStats,
) -> Result<(), LerobotError> {
    if stats.count == 0 {
        // A zero-length episode has no values to describe; keep the source
        // stats row verbatim rather than inventing degenerate values.
        return Ok(());
    }
    let count = i64::try_from(stats.count).map_err(|_| LerobotError::InconsistentMetadata {
        detail: format!("episode frame count {} exceeds i64 range", stats.count),
    })?;
    let cells = [
        ("min", StatScalar::Int(stats.minimum)),
        ("max", StatScalar::Int(stats.maximum)),
        ("mean", StatScalar::Float(stats.mean())),
        ("std", StatScalar::Float(stats.population_std())),
        ("count", StatScalar::Int(count)),
    ];
    for (aggregate_name, value) in cells {
        set_stat_list_value_if_present(
            schema,
            columns,
            &format!("stats/{stats_column_name}/{aggregate_name}"),
            value,
        )?;
    }
    Ok(())
}

/// A recomputed statistic destined for a single-element list stats cell.
#[derive(Debug, Clone, Copy)]
enum StatScalar {
    Int(i64),
    Float(f64),
}

/// Replaces `column_name` (if the schema has it) with a one-row,
/// single-element list holding `value`, built with the schema's own element
/// field so the exported schema stays byte-identical to the source's.
fn set_stat_list_value_if_present(
    schema: &SchemaRef,
    columns: &mut [ArrayRef],
    column_name: &str,
    value: StatScalar,
) -> Result<(), LerobotError> {
    let Ok(position) = require_field_position(schema, column_name) else {
        return Ok(());
    };
    let DataType::List(element_field) = schema.field(position).data_type() else {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!("stats column {column_name:?} is not a list"),
        });
    };

    let single_element_list: ListArray = match (element_field.data_type(), value) {
        (DataType::Int64, StatScalar::Int(int_value)) => {
            let mut builder =
                ListBuilder::new(Int64Builder::new()).with_field(element_field.clone());
            builder.values().append_value(int_value);
            builder.append(true);
            builder.finish()
        }
        (DataType::Float64, StatScalar::Int(int_value)) => {
            // Counts up to 2^53 frames round-trip losslessly through f64.
            #[allow(clippy::cast_precision_loss)]
            let float_value = int_value as f64;
            single_element_float64_list(element_field.clone(), float_value)
        }
        (DataType::Float64, StatScalar::Float(float_value)) => {
            single_element_float64_list(element_field.clone(), float_value)
        }
        (DataType::Int64, StatScalar::Float(_)) => {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "stats column {column_name:?} stores int64 elements but the recomputed \
                     statistic is fractional"
                ),
            });
        }
        (unsupported_element_type, _) => {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "stats column {column_name:?} has unsupported element type \
                     {unsupported_element_type:?}"
                ),
            });
        }
    };
    columns[position] = Arc::new(single_element_list) as ArrayRef;
    Ok(())
}

fn single_element_float64_list(element_field: arrow_schema::FieldRef, value: f64) -> ListArray {
    let mut builder = ListBuilder::new(Float64Builder::new()).with_field(element_field);
    builder.values().append_value(value);
    builder.append(true);
    builder.finish()
}

/// Writes the assembled subset's `meta/info.json` after every range landed:
/// the SOURCE document with totals patched and non-retained video features
/// stripped — the finalize counterpart of
/// [`export_episode_range_preserving_packed_videos`], which deliberately
/// skips info.
///
/// # Errors
///
/// Returns [`LerobotError`] when a retained key is not a source video
/// feature or the info document cannot be written.
pub fn write_packed_subset_info_json(
    src: &Dataset,
    episode_count: usize,
    total_frames: u64,
    task_count: usize,
    retained_video_keys: &[String],
    dest_root: &Path,
) -> Result<(), LerobotError> {
    write_packed_subset_info_json_with_extension(
        src,
        episode_count,
        total_frames,
        task_count,
        retained_video_keys,
        dest_root,
        None,
    )
}

/// Variant of [`write_packed_subset_info_json`] that applies an optional
/// metadata extension.
///
/// # Errors
///
/// Returns the same errors as [`write_packed_subset_info_json`].
pub fn write_packed_subset_info_json_with_extension(
    src: &Dataset,
    episode_count: usize,
    total_frames: u64,
    task_count: usize,
    retained_video_keys: &[String],
    dest_root: &Path,
    export_extension: Option<&dyn DatasetExportExtension>,
) -> Result<(), LerobotError> {
    let retained_video_keys = RetainedVideoKeys::parse(src, retained_video_keys)?;
    write_export_info_json(
        src,
        episode_count,
        total_frames,
        task_count,
        ExportedVideoLayout::PreservedPacked(&retained_video_keys),
        dest_root,
        export_extension,
    )
}

fn write_export_info_json(
    src: &Dataset,
    episode_count: usize,
    total_frames: u64,
    task_count: usize,
    exported_video_layout: ExportedVideoLayout<'_>,
    dest_root: &Path,
    export_extension: Option<&dyn DatasetExportExtension>,
) -> Result<(), LerobotError> {
    // Mutate a clone of the raw source document so fields this crate does not
    // model (features, robot_type, file size limits, future fields) survive.
    let mut document = src.info().raw_document().clone();
    let document_object =
        document
            .as_object_mut()
            .ok_or_else(|| LerobotError::InconsistentMetadata {
                detail: "source info.json root is not a JSON object".to_owned(),
            })?;
    document_object.insert("total_episodes".to_owned(), Value::from(episode_count));
    document_object.insert("total_frames".to_owned(), Value::from(total_frames));
    document_object.insert("total_tasks".to_owned(), Value::from(task_count));
    document_object.insert(
        "splits".to_owned(),
        serde_json::json!({ "train": format!("0:{episode_count}") }),
    );

    let features_object = document_object
        .get_mut("features")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: "source info.json has no features object".to_owned(),
        })?;
    match exported_video_layout {
        // Re-encoded video declarations must describe what was actually
        // written instead of the source encoding. A re-encoding export writes
        // only the views the video writer produced (the RGB surface); any
        // other video modality (depth, unclassified) has neither files nor
        // episode segment columns in the export, so its declaration is
        // removed too — a dangling declaration would make the export
        // unreadable.
        ExportedVideoLayout::Reencoded(camera_write_results) => {
            for source_video_key in src.info().video_keys() {
                if !camera_write_results.contains_key(&source_video_key) {
                    features_object.remove(&source_video_key);
                }
            }
            for (camera, write_result) in camera_write_results {
                patch_video_feature_declaration(
                    features_object,
                    camera,
                    write_result.written_video(),
                )?;
            }
        }
        // Packed copies preserve retained feature declarations byte-for-byte;
        // only video modalities excluded by the plan are removed.
        ExportedVideoLayout::PreservedPacked(retained_video_keys) => {
            for source_video_key in src.info().video_keys() {
                if !retained_video_keys.as_set().contains(&source_video_key) {
                    features_object.remove(&source_video_key);
                }
            }
        }
    }
    if let Some(export_extension) = export_extension {
        export_extension.rewrite_exported_info_document(document_object);
    }

    let info_json_text =
        serde_json::to_string_pretty(&document).map_err(|source| LerobotError::Json {
            context: "export info.json".to_owned(),
            source,
        })?;
    // Writer-owned directory creation: callers may hand a freshly created
    // destination root (the prepared finalize step does), so `meta/` cannot be
    // assumed to exist.
    let meta_dir = dest_root.join("meta");
    fs::create_dir_all(&meta_dir).map_err(|source| LerobotError::Io {
        path: meta_dir.clone(),
        source,
    })?;
    let dest_path = meta_dir.join("info.json");
    fs::write(&dest_path, info_json_text).map_err(|source| LerobotError::Io {
        path: dest_path,
        source,
    })
}

/// Rewrites one video feature's declaration to the written codec, resolution,
/// and `[height, width, 3]` shape. Keys match the real `LeRobot` layout:
/// `shape` on the feature itself, `video.codec` / `video.width` /
/// `video.height` inside its `info` object.
///
/// `names` must be rewritten together with `shape`: a channel-first source
/// declares `names` like `["channels", "height", "width"]`, and keeping that
/// list against the written channel-last shape would make
/// [`crate::FeatureSpec::video_kind`] read `shape[0]` as the channel count on
/// reopen, which would demote the camera to unknown and may violate an
/// extension's camera invariants.
fn patch_video_feature_declaration(
    features_object: &mut serde_json::Map<String, Value>,
    camera: &str,
    written_video: &WrittenVideo,
) -> Result<(), LerobotError> {
    let camera_feature = features_object
        .get_mut(camera)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: format!("source info.json has no feature object for video key {camera:?}"),
        })?;
    camera_feature.insert(
        "shape".to_owned(),
        serde_json::json!([written_video.height(), written_video.width(), 3]),
    );
    camera_feature.insert(
        "names".to_owned(),
        serde_json::json!(["height", "width", "channels"]),
    );
    let video_info_object = camera_feature
        .entry("info")
        .or_insert_with(|| Value::Object(serde_json::Map::new()))
        .as_object_mut()
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: format!("info.json feature {camera:?} has a non-object \"info\" entry"),
        })?;
    video_info_object.insert("video.codec".to_owned(), Value::from(written_video.codec()));
    video_info_object.insert("video.width".to_owned(), Value::from(written_video.width()));
    video_info_object.insert(
        "video.height".to_owned(),
        Value::from(written_video.height()),
    );
    Ok(())
}

fn copy_stats_json_if_present(source_root: &Path, dest_root: &Path) -> Result<(), LerobotError> {
    let source_stats = source_root.join("meta").join("stats.json");
    if !source_stats.is_file() {
        return Ok(());
    }
    // Aggregate stats describe the FULL source dataset, so they are only an
    // approximation for the subset — copied verbatim to keep python tooling
    // that expects the file working.
    let dest_stats = dest_root.join("meta").join("stats.json");
    fs::copy(&source_stats, &dest_stats).map_err(|source| LerobotError::Io {
        path: dest_stats,
        source,
    })?;
    Ok(())
}

fn create_arrow_writer(
    dest_path: &Path,
    schema: SchemaRef,
) -> Result<ArrowWriter<File>, LerobotError> {
    let dest_file = File::create(dest_path).map_err(|source| LerobotError::Io {
        path: dest_path.to_path_buf(),
        source,
    })?;
    // Level 1 preserves Snappy-like write throughput while reducing durable
    // export size. Parquet readers discover the codec from file metadata.
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    ArrowWriter::try_new(dest_file, schema, Some(properties)).map_err(|source| {
        LerobotError::Parquet {
            path: dest_path.to_path_buf(),
            source,
        }
    })
}

fn set_int64_value(
    schema: &SchemaRef,
    columns: &mut [ArrayRef],
    column_name: &str,
    value: i64,
) -> Result<(), LerobotError> {
    let position = require_field_position(schema, column_name)?;
    columns[position] = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
    Ok(())
}

fn set_int64_value_if_present(
    schema: &SchemaRef,
    columns: &mut [ArrayRef],
    column_name: &str,
    value: i64,
) {
    if let Ok(position) = require_field_position(schema, column_name) {
        columns[position] = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
    }
}

fn set_float64_value(
    schema: &SchemaRef,
    columns: &mut [ArrayRef],
    column_name: &str,
    value: f64,
) -> Result<(), LerobotError> {
    let position = require_field_position(schema, column_name)?;
    columns[position] = Arc::new(Float64Array::from(vec![value])) as ArrayRef;
    Ok(())
}

fn create_dir_all_with_context(dir: &Path) -> Result<(), LerobotError> {
    fs::create_dir_all(dir).map_err(|source| LerobotError::Io {
        path: dir.to_path_buf(),
        source,
    })
}

fn i64_from_usize(value: usize) -> i64 {
    i64::try_from(value).expect("row and task counts fit in i64")
}

fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).expect("row counts fit in u64")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_episodes_enforces_non_empty_unique_ordered_selections() {
        assert!(ExportEpisodes::try_from(Vec::new()).is_err());
        assert!(ExportEpisodes::try_from(vec![4, 4]).is_err());
        assert_eq!(
            ExportEpisodes::try_from(vec![4, 2])
                .expect("non-empty selection")
                .iter()
                .collect::<Vec<_>>(),
            vec![4, 2]
        );
    }

    #[test]
    fn contiguous_runs_finds_all_runs() {
        let column = Int64Array::from(vec![
            Some(1),
            Some(1),
            Some(2),
            Some(1),
            None,
            Some(1),
            Some(1),
        ]);
        assert_eq!(
            contiguous_runs_of_value(&column, 1),
            vec![(0, 2), (3, 1), (5, 2)]
        );
        assert_eq!(contiguous_runs_of_value(&column, 2), vec![(2, 1)]);
        assert_eq!(
            contiguous_runs_of_value(&column, 9),
            Vec::<(usize, usize)>::new()
        );
    }

    #[test]
    fn task_remap_assigns_indices_in_first_appearance_order() {
        let mut source_tasks = HashMap::new();
        source_tasks.insert(0, "pick".to_owned());
        source_tasks.insert(1, "place".to_owned());
        source_tasks.insert(7, "wave".to_owned());
        let mut remap = TaskRemap::new(source_tasks);

        assert_eq!(remap.remap(7).unwrap(), 0);
        assert_eq!(remap.remap(0).unwrap(), 1);
        assert_eq!(remap.remap(7).unwrap(), 0);
        assert_eq!(remap.remap(1).unwrap(), 2);
        assert_eq!(
            remap.new_task_strings,
            vec!["wave".to_owned(), "pick".to_owned(), "place".to_owned()]
        );
    }

    #[test]
    fn task_remap_rejects_unknown_source_index() {
        let mut remap = TaskRemap::new(HashMap::new());
        let error = remap.remap(3).unwrap_err();
        assert!(matches!(error, LerobotError::InconsistentMetadata { .. }));
    }

    /// The fixture's episode 0 spans global indices 0..=302; python lerobot
    /// records mean 151.0 and population std 87.46808941932291 for it, so the
    /// recomputed statistics must reproduce those exact aggregates.
    #[test]
    fn rewritten_column_stats_match_python_lerobot_population_aggregates() {
        let mut stats = RewrittenColumnStats::default();
        for value in 0..303 {
            stats.record(value);
        }
        assert_eq!(stats.minimum, 0);
        assert_eq!(stats.maximum, 302);
        assert_eq!(stats.count, 303);
        assert!((stats.mean() - 151.0).abs() < 1e-9);
        assert!((stats.population_std() - 87.468_089_419_322_91).abs() < 1e-9);
    }

    #[test]
    fn rewritten_column_stats_of_a_constant_column_have_zero_std() {
        let mut stats = RewrittenColumnStats::default();
        for _ in 0..10 {
            stats.record(7);
        }
        assert_eq!((stats.minimum, stats.maximum, stats.count), (7, 7, 10));
        assert!((stats.mean() - 7.0).abs() < f64::EPSILON);
        assert!(stats.population_std().abs() < f64::EPSILON);
    }

    /// Large baselines must not destroy precision: indices near 2^40 with a
    /// small spread still yield the exact spread statistics.
    #[test]
    fn rewritten_column_stats_stay_stable_for_large_global_indices() {
        let baseline = 1_i64 << 40;
        let mut stats = RewrittenColumnStats::default();
        for offset in 0..303 {
            stats.record(baseline + offset);
        }
        #[allow(clippy::cast_precision_loss)]
        let expected_mean = baseline as f64 + 151.0;
        assert!((stats.mean() - expected_mean).abs() < 1e-6);
        assert!((stats.population_std() - 87.468_089_419_322_91).abs() < 1e-9);
    }

    fn expect_destination_problem(
        claim_result: Result<ExportDestinationClaim, LerobotError>,
        expected_problem: ExportDestinationProblem,
    ) {
        match claim_result {
            Err(LerobotError::ExportDestination { problem, .. }) => {
                assert_eq!(problem, expected_problem);
            }
            Err(other_error) => panic!("expected ExportDestination error, got {other_error}"),
            Ok(_) => panic!("expected ExportDestination error, got a successful claim"),
        }
    }

    #[test]
    fn claim_rejects_destination_equal_to_source() {
        let source_dir = tempfile::tempdir().unwrap();
        expect_destination_problem(
            claim_export_destination(source_dir.path(), source_dir.path()),
            ExportDestinationProblem::SameAsSource,
        );
    }

    #[test]
    fn claim_rejects_destination_equal_to_source_via_dot_dot_segments() {
        let source_dir = tempfile::tempdir().unwrap();
        let dest_through_nonexistent_child = source_dir.path().join("nonexistent").join("..");
        expect_destination_problem(
            claim_export_destination(source_dir.path(), &dest_through_nonexistent_child),
            ExportDestinationProblem::SameAsSource,
        );
    }

    #[test]
    fn claim_rejects_destination_inside_source() {
        let source_dir = tempfile::tempdir().unwrap();
        let destination_inside_source = source_dir.path().join("exported");
        expect_destination_problem(
            claim_export_destination(source_dir.path(), &destination_inside_source),
            ExportDestinationProblem::InsideSource,
        );
    }

    #[test]
    fn claim_rejects_destination_containing_source() {
        let parent_dir = tempfile::tempdir().unwrap();
        let source_root = parent_dir.path().join("dataset");
        fs::create_dir(&source_root).unwrap();
        expect_destination_problem(
            claim_export_destination(&source_root, parent_dir.path()),
            ExportDestinationProblem::ContainsSource,
        );
    }

    #[test]
    fn claim_rejects_destination_that_is_a_file() {
        let source_dir = tempfile::tempdir().unwrap();
        let scratch_dir = tempfile::tempdir().unwrap();
        let file_destination = scratch_dir.path().join("occupied.txt");
        fs::write(&file_destination, b"occupied").unwrap();
        expect_destination_problem(
            claim_export_destination(source_dir.path(), &file_destination),
            ExportDestinationProblem::NotADirectory,
        );
    }

    #[test]
    fn claim_rejects_non_empty_destination_directory() {
        let source_dir = tempfile::tempdir().unwrap();
        let dest_dir = tempfile::tempdir().unwrap();
        fs::write(dest_dir.path().join("leftover.txt"), b"leftover").unwrap();
        expect_destination_problem(
            claim_export_destination(source_dir.path(), dest_dir.path()),
            ExportDestinationProblem::NotEmpty,
        );
    }

    #[test]
    fn claim_creates_missing_destination_and_removes_marker_on_drop() {
        let source_dir = tempfile::tempdir().unwrap();
        let scratch_dir = tempfile::tempdir().unwrap();
        let destination = scratch_dir.path().join("new").join("exported");

        let claim = claim_export_destination(source_dir.path(), &destination).unwrap();
        let marker_path = destination.join(EXPORT_CLAIM_MARKER_FILE_NAME);
        assert!(marker_path.is_file(), "claim must create the marker file");

        drop(claim);
        assert!(
            !marker_path.exists(),
            "dropping the claim must remove the marker file"
        );
        assert!(destination.is_dir(), "the claimed directory itself remains");
    }

    #[test]
    fn claim_preserves_a_coordinator_staging_ownership_manifest() {
        let source_dir = tempfile::tempdir().unwrap();
        let destination = tempfile::tempdir().unwrap();
        let ownership_manifest_path = destination
            .path()
            .join(EXPORT_STAGING_OWNERSHIP_MANIFEST_FILE_NAME);
        fs::write(&ownership_manifest_path, b"coordinator-owned").unwrap();

        let claim = claim_export_destination(source_dir.path(), destination.path())
            .expect("the reserved staging ownership proof is not dataset content");

        assert!(ownership_manifest_path.is_file());
        drop(claim);
        assert!(ownership_manifest_path.is_file());
    }

    #[test]
    fn second_concurrent_claim_fails_cleanly_and_destination_is_reusable_after_release() {
        let source_dir = tempfile::tempdir().unwrap();
        let dest_dir = tempfile::tempdir().unwrap();

        let first_claim = claim_export_destination(source_dir.path(), dest_dir.path()).unwrap();
        expect_destination_problem(
            claim_export_destination(source_dir.path(), dest_dir.path()),
            ExportDestinationProblem::AlreadyClaimed,
        );

        drop(first_claim);
        let reclaimed = claim_export_destination(source_dir.path(), dest_dir.path());
        assert!(
            reclaimed.is_ok(),
            "releasing a claim must make the destination claimable again"
        );
    }

    #[test]
    fn resolve_destination_root_folds_dot_dot_through_nonexistent_components() {
        let scratch_dir = tempfile::tempdir().unwrap();
        let canonical_scratch = fs::canonicalize(scratch_dir.path()).unwrap();
        let convoluted = scratch_dir
            .path()
            .join("missing-a")
            .join("missing-b")
            .join("..")
            .join(".")
            .join("final");
        assert_eq!(
            resolve_destination_root(&convoluted).unwrap(),
            canonical_scratch.join("missing-a").join("final")
        );
    }
}
