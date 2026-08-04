//! The single canonical reader for an episode's per-frame vector columns
//! (`observation.state`, `action`, …) out of the dataset's data parquet.
//!
//! Dataset consumers share this decode so row filtering, Arrow list handling,
//! and data-path rendering cannot drift.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
    LargeListArray, ListArray, RecordBatch, RecordBatchReader,
};
use arrow_schema::ArrowError;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder, RowFilter,
    RowSelection,
};

use crate::{Dataset, LerobotError};

/// The bookkeeping column that identifies which parquet rows belong to an
/// episode (a data file concatenates many episodes' rows in v3 datasets).
const EPISODE_INDEX_COLUMN: &str = "episode_index";
const FRAME_INDEX_COLUMN: &str = "frame_index";
const TIMESTAMP_COLUMN: &str = "timestamp";

/// One selected episode's expected rows inside a packed data parquet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedEpisodeDataRows {
    episode_index: u32,
    expected_row_count: usize,
    /// Exact file-local rows when metadata proves the file's episodes are
    /// contiguous. `None` keeps unusual-but-readable datasets on the predicate
    /// fallback instead of rejecting them.
    file_row_range: Option<Range<usize>>,
}

impl PlannedEpisodeDataRows {
    /// Creates one episode's row expectation.
    ///
    /// `file_row_range` is omitted when the metadata cannot prove an exact,
    /// contiguous file-local range and the reader must filter by episode ID.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidEpisodeDataReadPlan`] when an exact range has a
    /// different length than `expected_row_count`.
    pub fn new(
        episode_index: u32,
        expected_row_count: usize,
        file_row_range: Option<Range<usize>>,
    ) -> Result<Self, InvalidEpisodeDataReadPlan> {
        if let Some(row_range) = &file_row_range
            && row_range.end.checked_sub(row_range.start) != Some(expected_row_count)
        {
            return Err(InvalidEpisodeDataReadPlan::new(format!(
                "episode {episode_index} expects {expected_row_count} rows but its exact range is \
                 {}..{}",
                row_range.start, row_range.end
            )));
        }
        Ok(Self {
            episode_index,
            expected_row_count,
            file_row_range,
        })
    }

    /// Source episode index selected from this packed file.
    #[must_use]
    pub const fn episode_index(&self) -> u32 {
        self.episode_index
    }

    /// Row count declared by the episode metadata.
    #[must_use]
    pub const fn expected_row_count(&self) -> usize {
        self.expected_row_count
    }

    /// Exact file-local rows when metadata proves they are contiguous.
    #[must_use]
    pub fn file_row_range(&self) -> Option<Range<usize>> {
        self.file_row_range.clone()
    }
}

/// A materialized packed data file and every selected episode it contains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeDataFileReadPlan {
    data_parquet_path: PathBuf,
    episodes: Vec<PlannedEpisodeDataRows>,
    expected_file_row_count: Option<usize>,
}

impl EpisodeDataFileReadPlan {
    /// Creates a validated read plan for one packed Parquet file.
    ///
    /// Exact ranges must be present for every selected episode or none of
    /// them. Exact ranges must also be ordered, non-overlapping, and bounded
    /// by `expected_file_row_count`. Plans without exact ranges use the
    /// episode-index predicate fallback.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidEpisodeDataReadPlan`] when the plan is empty, repeats
    /// an episode, or carries inconsistent exact-range metadata.
    pub fn new(
        data_parquet_path: PathBuf,
        episodes: Vec<PlannedEpisodeDataRows>,
        expected_file_row_count: Option<usize>,
    ) -> Result<Self, InvalidEpisodeDataReadPlan> {
        if episodes.is_empty() {
            return Err(InvalidEpisodeDataReadPlan::new(
                "a packed-file read plan must select at least one episode",
            ));
        }

        let mut seen_episode_indices = HashSet::with_capacity(episodes.len());
        for episode in &episodes {
            if !seen_episode_indices.insert(episode.episode_index) {
                return Err(InvalidEpisodeDataReadPlan::new(format!(
                    "episode {} is selected more than once",
                    episode.episode_index
                )));
            }
        }

        let exact_range_count = episodes
            .iter()
            .filter(|episode| episode.file_row_range.is_some())
            .count();
        if exact_range_count != 0 && exact_range_count != episodes.len() {
            return Err(InvalidEpisodeDataReadPlan::new(
                "exact file-local ranges must be present for every selected episode or none",
            ));
        }
        if exact_range_count == 0 && expected_file_row_count.is_some() {
            return Err(InvalidEpisodeDataReadPlan::new(
                "an expected packed-file row count requires exact episode row ranges",
            ));
        }
        if exact_range_count == episodes.len() {
            let expected_file_row_count = expected_file_row_count.ok_or_else(|| {
                InvalidEpisodeDataReadPlan::new(
                    "exact episode row ranges require the expected packed-file row count",
                )
            })?;
            let mut previous_range_end = 0;
            for episode in &episodes {
                let row_range = episode
                    .file_row_range
                    .as_ref()
                    .expect("every exact range was established above");
                if row_range.start < previous_range_end {
                    return Err(InvalidEpisodeDataReadPlan::new(format!(
                        "episode {} range {}..{} overlaps or precedes an earlier range",
                        episode.episode_index, row_range.start, row_range.end
                    )));
                }
                if row_range.end > expected_file_row_count {
                    return Err(InvalidEpisodeDataReadPlan::new(format!(
                        "episode {} range {}..{} exceeds the expected file row count {expected_file_row_count}",
                        episode.episode_index, row_range.start, row_range.end
                    )));
                }
                previous_range_end = row_range.end;
            }
        }

        Ok(Self {
            data_parquet_path,
            episodes,
            expected_file_row_count,
        })
    }

    /// Local packed Parquet path prepared for synchronous reading.
    #[must_use]
    pub fn data_parquet_path(&self) -> &Path {
        &self.data_parquet_path
    }

    /// Selected episode row expectations within this file.
    #[must_use]
    pub fn episodes(&self) -> &[PlannedEpisodeDataRows] {
        &self.episodes
    }

    /// Complete packed-file row count when metadata establishes it.
    #[must_use]
    pub const fn expected_file_row_count(&self) -> Option<usize> {
        self.expected_file_row_count
    }
}

/// A packed-file read plan whose row claims are internally inconsistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidEpisodeDataReadPlan {
    detail: String,
}

impl InvalidEpisodeDataReadPlan {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// Human-readable reason the plan was rejected.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for InvalidEpisodeDataReadPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl std::error::Error for InvalidEpisodeDataReadPlan {}

/// Complete read plan for a requested episode set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeDataReadPlan {
    data_files: Vec<EpisodeDataFileReadPlan>,
    missing_episode_indices: Vec<u32>,
    prefetched_file_count: u64,
}

impl EpisodeDataReadPlan {
    /// Borrows the prepared packed-file plans.
    #[must_use]
    pub fn data_files(&self) -> &[EpisodeDataFileReadPlan] {
        &self.data_files
    }

    /// Requested episode indices absent from the source metadata.
    #[must_use]
    pub fn missing_episode_indices(&self) -> &[u32] {
        &self.missing_episode_indices
    }

    /// Number of distinct remote files materialized for this plan.
    #[must_use]
    pub const fn prefetched_file_count(&self) -> u64 {
        self.prefetched_file_count
    }

    /// Consumes the plan for concurrent execution without exposing mutable
    /// invariant-bearing fields.
    #[must_use]
    pub fn into_parts(self) -> (Vec<EpisodeDataFileReadPlan>, Vec<u32>, u64) {
        (
            self.data_files,
            self.missing_episode_indices,
            self.prefetched_file_count,
        )
    }
}

/// How one packed parquet was narrowed to the selected episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeVectorSelectionMode {
    /// Scan every row because the entire file belongs to selected episodes.
    FullFile,
    /// Prune the file to metadata-proven contiguous row ranges.
    ExactRowRanges,
    /// Scan candidate rows and retain those with selected episode indices.
    EpisodeIndexPredicate,
}

/// Projected source-column availability for a streaming vector read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeVectorReadSchema {
    present_column_names: HashSet<String>,
    has_frame_index: bool,
    has_timestamp: bool,
}

impl EpisodeVectorReadSchema {
    /// Whether the source parquet declares `column_name`.
    #[must_use]
    pub fn column_is_present(&self, column_name: &str) -> bool {
        self.present_column_names.contains(column_name)
    }

    /// Whether decoded rows include the source `frame_index` column.
    #[must_use]
    pub fn has_frame_index(&self) -> bool {
        self.has_frame_index
    }

    /// Whether decoded rows include the source `timestamp` column.
    #[must_use]
    pub fn has_timestamp(&self) -> bool {
        self.has_timestamp
    }
}

/// One full-resolution row yielded by [`EpisodeVectorRowReader`].
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeVectorRow {
    /// Position of the episode in the caller's requested episode order.
    pub episode_position: usize,
    /// Episode index declared by the dataset.
    pub episode_index: u32,
    /// Zero-based decoded row position within the episode.
    pub frame_position: usize,
    /// Source frame index when that bookkeeping column is present.
    pub frame_index: Option<i64>,
    /// Source timestamp when that bookkeeping column is present.
    pub timestamp: Option<f64>,
    /// Requested vector columns in request order. A source-missing column has
    /// empty rows and is distinguished through [`EpisodeVectorReadSchema`].
    pub vectors: Vec<Vec<f32>>,
}

/// Observable work performed by one packed-file reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodeVectorReadReport {
    /// Strategy used to select episode rows from the packed file.
    pub selection_mode: EpisodeVectorSelectionMode,
    /// Total rows described by the source parquet metadata.
    pub source_row_count: usize,
    /// Rows expected after applying the selection strategy.
    pub selected_row_count: usize,
    /// Rows decoded and yielded so far.
    pub decoded_row_count: usize,
}

/// Streaming reader that decodes selected packed-file rows exactly once.
pub struct EpisodeVectorRowReader {
    data_parquet_path: PathBuf,
    reader: ParquetRecordBatchReader,
    current_batch: Option<RecordBatch>,
    current_row_position: usize,
    requested_column_positions: Vec<Option<usize>>,
    episode_index_column_position: usize,
    frame_index_column_position: Option<usize>,
    timestamp_column_position: Option<usize>,
    episode_position_by_index: HashMap<i64, usize>,
    expected_row_counts: Vec<usize>,
    decoded_row_counts: Vec<usize>,
    schema: EpisodeVectorReadSchema,
    report: EpisodeVectorReadReport,
    completion_checked: bool,
}

/// One requested vector column's decoded per-frame rows.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeVectorColumn {
    /// The column name exactly as requested (e.g. `observation.state`).
    pub name: String,
    /// One `Vec<f32>` per frame row of the episode, in frame order. A null list
    /// element decodes to `0.0`. Empty when the column is absent from the data
    /// parquet (a dataset that does not carry it), so callers render such a
    /// dataset gracefully rather than erroring.
    pub rows: Vec<Vec<f32>>,
}

/// One episode's decoded per-frame vector columns, in the order they were
/// requested.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeVectorColumns {
    columns: Vec<EpisodeVectorColumn>,
    present_column_names: HashSet<String>,
    frame_indices: Option<Vec<i64>>,
    timestamps: Option<Vec<f64>>,
}

impl EpisodeVectorColumns {
    /// The decoded frame rows of a requested column, or `None` when the column
    /// was not requested. A requested column that is absent from the parquet
    /// returns `Some(&[])`.
    #[must_use]
    pub fn column(&self, name: &str) -> Option<&[Vec<f32>]> {
        self.columns
            .iter()
            .find(|column| column.name == name)
            .map(|column| column.rows.as_slice())
    }

    /// Every requested column, in request order.
    #[must_use]
    pub fn columns(&self) -> &[EpisodeVectorColumn] {
        &self.columns
    }

    /// Whether the source parquet declares a requested vector column. This
    /// distinguishes a missing input from a present column with zero rows.
    #[must_use]
    pub fn column_is_present(&self, name: &str) -> bool {
        self.present_column_names.contains(name)
    }

    /// Raw per-row frame indices, or `None` when the source parquet does not
    /// declare `frame_index`.
    #[must_use]
    pub fn frame_indices(&self) -> Option<&[i64]> {
        self.frame_indices.as_deref()
    }

    /// Raw per-row timestamps, promoted to f64, or `None` when the source
    /// parquet does not declare `timestamp`.
    #[must_use]
    pub fn timestamps(&self) -> Option<&[f64]> {
        self.timestamps.as_deref()
    }
}

impl Dataset {
    /// Materializes and groups selected episodes by their packed data parquet.
    ///
    /// Metadata that proves file-local contiguous row ranges is retained for
    /// page pruning. Datasets with unusual file ordering remain readable via
    /// the episode-index predicate fallback.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError`] when a data path cannot be rendered or a
    /// selected data parquet cannot be materialized.
    pub async fn plan_episode_data_reads(
        &self,
        episode_indices: &[u32],
    ) -> Result<EpisodeDataReadPlan, LerobotError> {
        let prefetched_file_count = self.prefetch_episode_data(episode_indices).await?;
        let requested_episode_indices: HashSet<u32> = episode_indices.iter().copied().collect();
        let missing_episode_indices: Vec<u32> = episode_indices
            .iter()
            .copied()
            .filter(|episode_index| self.episode(*episode_index).is_none())
            .collect();

        let mut all_episodes_by_relative_path: BTreeMap<String, Vec<&crate::Episode>> =
            BTreeMap::new();
        for episode in self.episodes() {
            let relative_path = self.episode_data_relative_key(episode.episode_index())?;
            all_episodes_by_relative_path
                .entry(relative_path)
                .or_default()
                .push(episode);
        }

        let mut data_files = Vec::new();
        for episodes_in_file in all_episodes_by_relative_path.values_mut() {
            if !episodes_in_file
                .iter()
                .any(|episode| requested_episode_indices.contains(&episode.episode_index()))
            {
                continue;
            }
            episodes_in_file.sort_unstable_by_key(|episode| episode.dataset_from_index());
            let file_base_row = episodes_in_file
                .first()
                .map_or(0, |episode| episode.dataset_from_index());
            let metadata_ranges_are_contiguous = episodes_in_file
                .windows(2)
                .all(|adjacent| adjacent[0].dataset_to_index() == adjacent[1].dataset_from_index());
            let expected_file_row_count = metadata_ranges_are_contiguous
                .then(|| {
                    episodes_in_file
                        .last()
                        .and_then(|episode| episode.dataset_to_index().checked_sub(file_base_row))
                        .and_then(|row_count| usize::try_from(row_count).ok())
                })
                .flatten();

            let selected_episodes: Vec<PlannedEpisodeDataRows> = episodes_in_file
                .iter()
                .filter(|episode| requested_episode_indices.contains(&episode.episode_index()))
                .map(|episode| {
                    let expected_row_count =
                        usize::try_from(episode.length_frames()).map_err(|_| {
                            LerobotError::InconsistentMetadata {
                                detail: format!(
                                    "episode {} frame count {} exceeds usize",
                                    episode.episode_index(),
                                    episode.length_frames()
                                ),
                            }
                        })?;
                    let file_row_range = if metadata_ranges_are_contiguous {
                        episode
                            .dataset_from_index()
                            .checked_sub(file_base_row)
                            .zip(episode.dataset_to_index().checked_sub(file_base_row))
                            .and_then(|(start, end)| {
                                Some(usize::try_from(start).ok()?..usize::try_from(end).ok()?)
                            })
                    } else {
                        None
                    };
                    PlannedEpisodeDataRows::new(
                        episode.episode_index(),
                        expected_row_count,
                        file_row_range,
                    )
                    .map_err(LerobotError::from)
                })
                .collect::<Result<_, LerobotError>>()?;
            let first_selected_episode = selected_episodes
                .first()
                .expect("a selected data file has at least one selected episode");
            let data_parquet_path =
                self.local_episode_data_parquet_path(first_selected_episode.episode_index)?;
            data_files.push(EpisodeDataFileReadPlan::new(
                data_parquet_path,
                selected_episodes,
                expected_file_row_count,
            )?);
        }

        Ok(EpisodeDataReadPlan {
            data_files,
            missing_episode_indices,
            prefetched_file_count,
        })
    }

    /// Reads one episode's per-frame vector columns from the dataset's data
    /// parquet, decoding each `Float32` list column (`List`, `LargeList`, or
    /// `FixedSizeList`) into `Vec<f32>` rows in frame order.
    ///
    /// Rows are matched by the `episode_index` COLUMN value, not by global row
    /// position (a data file need not begin at global row 0). A requested column
    /// absent from the parquet yields an empty [`EpisodeVectorColumn::rows`]
    /// (not an error), so a dataset lacking `action` or `observation.state`
    /// still reads cleanly.
    ///
    /// The download of an object-store data file is awaited, and the CPU-bound
    /// parquet scan of the resulting local file runs on a blocking thread.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::EpisodeNotFound`] for an unknown episode index,
    /// [`LerobotError::Io`] when the data parquet cannot be opened,
    /// [`LerobotError::Parquet`] / [`LerobotError::Arrow`] when it cannot be
    /// read or decoded, and [`LerobotError::InconsistentMetadata`] when a
    /// selected column is not a float32 list or the `episode_index` column is
    /// missing or not int64.
    pub async fn read_episode_vector_columns(
        &self,
        episode_index: u32,
        columns: &[&str],
    ) -> Result<EpisodeVectorColumns, LerobotError> {
        let data_parquet_path = self.episode_data_parquet_path(episode_index).await?;
        let task_data_parquet_path = data_parquet_path.clone();
        let owned_requested_columns: Vec<String> = columns
            .iter()
            .map(|column_name| (*column_name).to_owned())
            .collect();
        tokio::task::spawn_blocking(move || {
            let requested_column_names: Vec<&str> =
                owned_requested_columns.iter().map(String::as_str).collect();
            let mut columns_by_episode = read_episodes_vector_columns_from_file(
                &data_parquet_path,
                &[episode_index],
                &requested_column_names,
            )?;
            Ok(columns_by_episode
                .remove(&episode_index)
                .unwrap_or_else(|| empty_columns(&requested_column_names, &[], false, false)))
        })
        .await
        .map_err(|join_error| LerobotError::Io {
            path: task_data_parquet_path,
            source: std::io::Error::other(format!(
                "episode vector parquet scan task failed: {join_error}"
            )),
        })?
    }
}

fn planned_episode_selection(
    plan: &EpisodeDataFileReadPlan,
    source_row_count: usize,
) -> (EpisodeVectorSelectionMode, usize) {
    let all_ranges_are_exact = plan.expected_file_row_count == Some(source_row_count)
        && plan
            .episodes
            .iter()
            .all(|episode| episode.file_row_range.is_some());
    let selected_row_count = plan
        .episodes
        .iter()
        .map(|episode| episode.expected_row_count)
        .sum();
    let selection_mode = if all_ranges_are_exact && selected_row_count == source_row_count {
        EpisodeVectorSelectionMode::FullFile
    } else if all_ranges_are_exact {
        EpisodeVectorSelectionMode::ExactRowRanges
    } else {
        EpisodeVectorSelectionMode::EpisodeIndexPredicate
    };
    (selection_mode, selected_row_count)
}

fn apply_planned_episode_selection(
    builder: ParquetRecordBatchReaderBuilder<File>,
    plan: &EpisodeDataFileReadPlan,
    source_row_count: usize,
    selection_mode: EpisodeVectorSelectionMode,
) -> ParquetRecordBatchReaderBuilder<File> {
    match selection_mode {
        EpisodeVectorSelectionMode::FullFile => builder,
        EpisodeVectorSelectionMode::ExactRowRanges => {
            let selected_ranges = plan
                .episodes
                .iter()
                .map(|episode| {
                    episode
                        .file_row_range
                        .clone()
                        .expect("exact selection has every row range")
                })
                .collect::<Vec<_>>();
            builder.with_row_selection(RowSelection::from_consecutive_ranges(
                selected_ranges.into_iter(),
                source_row_count,
            ))
        }
        EpisodeVectorSelectionMode::EpisodeIndexPredicate => {
            let selected_episode_indices = plan
                .episodes
                .iter()
                .map(|episode| episode.episode_index)
                .collect::<Vec<_>>();
            let row_filter =
                episode_row_filter(builder.parquet_schema(), &selected_episode_indices);
            builder.with_row_filter(row_filter)
        }
    }
}

struct ProjectedColumnPositions {
    requested: Vec<Option<usize>>,
    episode_index: usize,
    frame_index: Option<usize>,
    timestamp: Option<usize>,
}

fn projected_column_positions(
    projected_schema: &arrow_schema::Schema,
    requested_columns: &[&str],
) -> ProjectedColumnPositions {
    ProjectedColumnPositions {
        requested: requested_columns
            .iter()
            .map(|column_name| {
                projected_schema
                    .column_with_name(column_name)
                    .map(|(position, _)| position)
            })
            .collect(),
        episode_index: projected_schema
            .column_with_name(EPISODE_INDEX_COLUMN)
            .map(|(position, _)| position)
            .expect("episode-index projection was required"),
        frame_index: projected_schema
            .column_with_name(FRAME_INDEX_COLUMN)
            .map(|(position, _)| position),
        timestamp: projected_schema
            .column_with_name(TIMESTAMP_COLUMN)
            .map(|(position, _)| position),
    }
}

impl EpisodeVectorRowReader {
    /// Opens one planned packed-file stream.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError`] when the parquet schema is invalid or the
    /// projected reader cannot be built.
    pub fn open(
        plan: &EpisodeDataFileReadPlan,
        requested_columns: &[&str],
    ) -> Result<Self, LerobotError> {
        let file = File::open(&plan.data_parquet_path).map_err(|source| LerobotError::Io {
            path: plan.data_parquet_path.clone(),
            source,
        })?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| {
            LerobotError::Parquet {
                path: plan.data_parquet_path.clone(),
                source,
            }
        })?;
        let source_row_count = usize::try_from(builder.metadata().file_metadata().num_rows())
            .map_err(|_| LerobotError::InconsistentMetadata {
                detail: format!(
                    "{} row count exceeds usize",
                    plan.data_parquet_path.display()
                ),
            })?;
        let source_schema = builder.schema();
        let has_frame_index = source_schema.column_with_name(FRAME_INDEX_COLUMN).is_some();
        let has_timestamp = source_schema.column_with_name(TIMESTAMP_COLUMN).is_some();
        let present_column_names: HashSet<String> = requested_columns
            .iter()
            .filter(|column_name| source_schema.column_with_name(column_name).is_some())
            .map(|column_name| (*column_name).to_owned())
            .collect();
        if source_schema
            .column_with_name(EPISODE_INDEX_COLUMN)
            .is_none()
        {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "{} has no {EPISODE_INDEX_COLUMN:?} column",
                    plan.data_parquet_path.display()
                ),
            });
        }

        let wanted_column_names: Vec<&str> = std::iter::once(EPISODE_INDEX_COLUMN)
            .chain(
                requested_columns
                    .iter()
                    .copied()
                    .filter(|name| present_column_names.contains(*name)),
            )
            .chain(has_frame_index.then_some(FRAME_INDEX_COLUMN))
            .chain(has_timestamp.then_some(TIMESTAMP_COLUMN))
            .collect();
        let projection = ProjectionMask::roots(
            builder.parquet_schema(),
            root_indices_for_columns(source_schema.as_ref(), &wanted_column_names),
        );
        let (selection_mode, selected_row_count) =
            planned_episode_selection(plan, source_row_count);
        let builder = builder.with_projection(projection);
        let builder =
            apply_planned_episode_selection(builder, plan, source_row_count, selection_mode);
        let reader = builder.build().map_err(|source| LerobotError::Parquet {
            path: plan.data_parquet_path.clone(),
            source,
        })?;
        let projected_schema = reader.schema();
        let column_positions =
            projected_column_positions(projected_schema.as_ref(), requested_columns);
        let episode_position_by_index = plan
            .episodes
            .iter()
            .enumerate()
            .map(|(episode_position, episode)| (i64::from(episode.episode_index), episode_position))
            .collect();
        let expected_row_counts = plan
            .episodes
            .iter()
            .map(|episode| episode.expected_row_count)
            .collect::<Vec<_>>();

        Ok(Self {
            data_parquet_path: plan.data_parquet_path.clone(),
            reader,
            current_batch: None,
            current_row_position: 0,
            requested_column_positions: column_positions.requested,
            episode_index_column_position: column_positions.episode_index,
            frame_index_column_position: column_positions.frame_index,
            timestamp_column_position: column_positions.timestamp,
            episode_position_by_index,
            decoded_row_counts: vec![0; plan.episodes.len()],
            expected_row_counts,
            schema: EpisodeVectorReadSchema {
                present_column_names,
                has_frame_index,
                has_timestamp,
            },
            report: EpisodeVectorReadReport {
                selection_mode,
                source_row_count,
                selected_row_count,
                decoded_row_count: 0,
            },
            completion_checked: false,
        })
    }

    /// Projected columns available to rows yielded by this reader.
    #[must_use]
    pub fn schema(&self) -> &EpisodeVectorReadSchema {
        &self.schema
    }

    /// Current selection and decoding statistics.
    #[must_use]
    pub fn report(&self) -> EpisodeVectorReadReport {
        self.report
    }

    fn completion_error(&self) -> Option<LerobotError> {
        self.expected_row_counts
            .iter()
            .zip(&self.decoded_row_counts)
            .enumerate()
            .find_map(
                |(episode_position, (&expected_row_count, &decoded_row_count))| {
                    (expected_row_count != decoded_row_count).then(|| {
                        let episode_index = self
                            .episode_position_by_index
                            .iter()
                            .find_map(|(&episode_index, &mapped_position)| {
                                (mapped_position == episode_position).then_some(episode_index)
                            })
                            .unwrap_or(-1);
                        LerobotError::InconsistentMetadata {
                            detail: format!(
                                "{} episode {episode_index} declares {expected_row_count} rows but \
                                 the parquet yielded {decoded_row_count}",
                                self.data_parquet_path.display()
                            ),
                        }
                    })
                },
            )
    }

    fn advance_to_available_row(&mut self) -> Result<bool, LerobotError> {
        while self
            .current_batch
            .as_ref()
            .is_none_or(|batch| self.current_row_position >= batch.num_rows())
        {
            match self.reader.next() {
                Some(Ok(batch)) => {
                    self.current_batch = Some(batch);
                    self.current_row_position = 0;
                }
                Some(Err(source)) => return Err(LerobotError::Arrow { source }),
                None => {
                    if self.completion_checked {
                        return Ok(false);
                    }
                    self.completion_checked = true;
                    if let Some(error) = self.completion_error() {
                        return Err(error);
                    }
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn decode_current_row(
        &self,
        row_position: usize,
    ) -> Result<DecodedEpisodeVectorValues, LerobotError> {
        let batch = self
            .current_batch
            .as_ref()
            .expect("current batch was populated");
        let episode_column = batch
            .column(self.episode_index_column_position)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| LerobotError::InconsistentMetadata {
                detail: format!(
                    "{} projected episode_index is not int64",
                    self.data_parquet_path.display()
                ),
            })?;
        if episode_column.is_null(row_position) {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "{} contains a null episode_index",
                    self.data_parquet_path.display()
                ),
            });
        }
        let raw_episode_index = episode_column.value(row_position);
        let episode_position = self
            .episode_position_by_index
            .get(&raw_episode_index)
            .copied()
            .ok_or_else(|| LerobotError::InconsistentMetadata {
                detail: format!(
                    "{} selected rows unexpectedly contain episode {raw_episode_index}",
                    self.data_parquet_path.display()
                ),
            })?;
        let episode_index =
            u32::try_from(raw_episode_index).map_err(|_| LerobotError::InconsistentMetadata {
                detail: format!(
                    "{} has out-of-range episode index {raw_episode_index}",
                    self.data_parquet_path.display()
                ),
            })?;
        let frame_index = self.frame_index_column_position.map(|column_position| {
            let column = batch
                .column(column_position)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("validated frame-index projection");
            if column.is_null(row_position) {
                i64::MIN
            } else {
                column.value(row_position)
            }
        });
        let timestamp = self.timestamp_column_position.map(|column_position| {
            timestamp_value(
                batch.column(column_position),
                row_position,
                &self.data_parquet_path,
            )
        });
        let timestamp = match timestamp {
            Some(Ok(timestamp)) => Some(timestamp),
            Some(Err(error)) => return Err(error),
            None => None,
        };
        let vectors = self
            .requested_column_positions
            .iter()
            .map(|column_position| {
                column_position.map_or_else(
                    || Ok(Vec::new()),
                    |column_position| {
                        decode_float32_list_row(
                            batch.column(column_position),
                            row_position,
                            batch.schema().field(column_position).name(),
                            &self.data_parquet_path,
                        )
                    },
                )
            })
            .collect::<Result<Vec<_>, LerobotError>>()?;
        Ok(DecodedEpisodeVectorValues {
            episode_position,
            episode_index,
            frame_index,
            timestamp,
            vectors,
        })
    }
}

struct DecodedEpisodeVectorValues {
    episode_position: usize,
    episode_index: u32,
    frame_index: Option<i64>,
    timestamp: Option<f64>,
    vectors: Vec<Vec<f32>>,
}

impl Iterator for EpisodeVectorRowReader {
    type Item = Result<EpisodeVectorRow, LerobotError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance_to_available_row() {
            Ok(true) => {}
            Ok(false) => return None,
            Err(error) => return Some(Err(error)),
        }

        let row_position = self.current_row_position;
        self.current_row_position += 1;
        let decoded_values = match self.decode_current_row(row_position) {
            Ok(decoded_values) => decoded_values,
            Err(error) => return Some(Err(error)),
        };
        let frame_position = self.decoded_row_counts[decoded_values.episode_position];
        self.decoded_row_counts[decoded_values.episode_position] += 1;
        self.report.decoded_row_count += 1;
        Some(Ok(EpisodeVectorRow {
            episode_position: decoded_values.episode_position,
            episode_index: decoded_values.episode_index,
            frame_position,
            frame_index: decoded_values.frame_index,
            timestamp: decoded_values.timestamp,
            vectors: decoded_values.vectors,
        }))
    }
}

/// Reads the requested vector columns for a SET of episodes out of a single
/// data parquet file in ONE pass, decoding each `Float32` list column (`List`,
/// `LargeList`, or `FixedSizeList`) into `Vec<f32>` rows in frame order.
///
/// A v3 data file concatenates MANY episodes' rows, so grouping episodes by
/// their data-parquet path ([`Dataset::episode_data_parquet_path`]) and calling
/// this once per file scans each file exactly ONCE — far cheaper than a full
/// scan per episode. Rows are matched by the `episode_index` COLUMN value, not
/// by global row position.
///
/// Every requested episode gets an entry in the returned map: an episode with
/// no matching rows — or a file that declares none of the requested columns —
/// maps to empty [`EpisodeVectorColumn::rows`] (not a missing key and not an
/// error), so callers can align results positionally.
///
/// BLOCKING: opens and scans a parquet file synchronously; async callers must
/// wrap this in `tokio::task::spawn_blocking` (or accept the block on a
/// batch/offline path).
///
/// # Errors
///
/// Returns [`LerobotError::Io`] when the data parquet cannot be opened,
/// [`LerobotError::Parquet`] / [`LerobotError::Arrow`] when it cannot be read
/// or decoded, and [`LerobotError::InconsistentMetadata`] when a selected
/// column is not a float32 list or the `episode_index` column is missing or not
/// int64.
// Keeping projection, filtering, and every aligned accumulator in this one
// pass makes it auditable that scalar diagnostics rows cannot drift from their
// vector rows.
#[allow(clippy::too_many_lines)]
pub fn read_episodes_vector_columns_from_file(
    data_parquet_path: &Path,
    episode_indices: &[u32],
    requested_columns: &[&str],
) -> Result<BTreeMap<u32, EpisodeVectorColumns>, LerobotError> {
    let file = File::open(data_parquet_path).map_err(|source| LerobotError::Io {
        path: data_parquet_path.to_path_buf(),
        source,
    })?;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| LerobotError::Parquet {
            path: data_parquet_path.to_path_buf(),
            source,
        })?;

    let has_frame_index = builder
        .schema()
        .column_with_name(FRAME_INDEX_COLUMN)
        .is_some();
    let has_timestamp = builder
        .schema()
        .column_with_name(TIMESTAMP_COLUMN)
        .is_some();

    // Only the requested vector columns the parquet actually declares are
    // decoded. The raw frame-index and timestamp columns travel with the same
    // full-resolution read when present so diagnostics cannot accidentally run
    // on a downsampled view.
    let present_columns: Vec<&str> = requested_columns
        .iter()
        .copied()
        .filter(|name| builder.schema().column_with_name(name).is_some())
        .collect();
    if present_columns.is_empty() && !has_frame_index && !has_timestamp {
        return Ok(episode_indices
            .iter()
            .map(|&episode_index| {
                (
                    episode_index,
                    empty_columns(requested_columns, &present_columns, false, false),
                )
            })
            .collect());
    }

    // Project to the episode filter, present vector columns, and available
    // diagnostics bookkeeping columns.
    let wanted_column_names: Vec<&str> = std::iter::once(EPISODE_INDEX_COLUMN)
        .chain(present_columns.iter().copied())
        .chain(has_frame_index.then_some(FRAME_INDEX_COLUMN))
        .chain(has_timestamp.then_some(TIMESTAMP_COLUMN))
        .collect();
    let wanted_root_indices =
        root_indices_for_columns(builder.schema().as_ref(), &wanted_column_names);
    let projection = ProjectionMask::roots(builder.parquet_schema(), wanted_root_indices);
    let episode_filter = episode_row_filter(builder.parquet_schema(), episode_indices);
    let reader = builder
        .with_projection(projection)
        .with_row_filter(episode_filter)
        .build()
        .map_err(|source| LerobotError::Parquet {
            path: data_parquet_path.to_path_buf(),
            source,
        })?;

    // One row accumulator per (requested episode, present column). Pre-populating
    // every requested episode means an episode with no rows still yields an
    // entry, and `get_mut` on the outer map doubles as the "this row belongs to
    // a requested episode" filter during the single pass.
    let mut rows_by_episode: BTreeMap<i64, BTreeMap<&str, Vec<Vec<f32>>>> = episode_indices
        .iter()
        .map(|&episode_index| {
            (
                i64::from(episode_index),
                present_columns
                    .iter()
                    .map(|name| (*name, Vec::new()))
                    .collect(),
            )
        })
        .collect();
    let mut frame_indices_by_episode: BTreeMap<i64, Vec<i64>> = episode_indices
        .iter()
        .map(|&episode_index| (i64::from(episode_index), Vec::new()))
        .collect();
    let mut timestamps_by_episode: BTreeMap<i64, Vec<f64>> = episode_indices
        .iter()
        .map(|&episode_index| (i64::from(episode_index), Vec::new()))
        .collect();
    for batch_result in reader {
        let batch = batch_result?;
        let episode_column = require_int64_column(&batch, EPISODE_INDEX_COLUMN, data_parquet_path)?;
        let frame_index_column = if has_frame_index {
            Some(require_int64_column(
                &batch,
                FRAME_INDEX_COLUMN,
                data_parquet_path,
            )?)
        } else {
            None
        };
        let timestamp_column = if has_timestamp {
            Some(require_timestamp_column(&batch, data_parquet_path)?)
        } else {
            None
        };
        for row in 0..batch.num_rows() {
            if episode_column.is_null(row) {
                continue;
            }
            let episode_index = episode_column.value(row);
            if let Some(frame_index_column) = frame_index_column
                && let Some(frame_indices) = frame_indices_by_episode.get_mut(&episode_index)
            {
                frame_indices.push(if frame_index_column.is_null(row) {
                    i64::MIN
                } else {
                    frame_index_column.value(row)
                });
            }
            if let Some(timestamp_column) = timestamp_column.as_ref()
                && let Some(timestamps) = timestamps_by_episode.get_mut(&episode_index)
            {
                timestamps.push(timestamp_column.value(row));
            }
        }
        for &column_name in &present_columns {
            let column = batch.column_by_name(column_name).ok_or_else(|| {
                LerobotError::InconsistentMetadata {
                    detail: format!(
                        "projected column {column_name:?} missing from a batch of {}",
                        data_parquet_path.display()
                    ),
                }
            })?;
            for row in 0..batch.num_rows() {
                if episode_column.is_null(row) {
                    continue;
                }
                let Some(per_column_rows) = rows_by_episode.get_mut(&episode_column.value(row))
                else {
                    continue;
                };
                let decoded_rows = per_column_rows
                    .get_mut(column_name)
                    .expect("present column has an accumulator");
                decoded_rows.push(decode_float32_list_row(
                    column,
                    row,
                    column_name,
                    data_parquet_path,
                )?);
            }
        }
    }

    Ok(episode_indices
        .iter()
        .map(|&episode_index| {
            let mut per_column_rows = rows_by_episode
                .remove(&i64::from(episode_index))
                .unwrap_or_default();
            let columns = requested_columns
                .iter()
                .map(|name| EpisodeVectorColumn {
                    name: (*name).to_owned(),
                    rows: per_column_rows.remove(name).unwrap_or_default(),
                })
                .collect();
            (
                episode_index,
                EpisodeVectorColumns {
                    columns,
                    present_column_names: present_columns
                        .iter()
                        .map(|column_name| (*column_name).to_owned())
                        .collect(),
                    frame_indices: has_frame_index.then(|| {
                        frame_indices_by_episode
                            .remove(&i64::from(episode_index))
                            .unwrap_or_default()
                    }),
                    timestamps: has_timestamp.then(|| {
                        timestamps_by_episode
                            .remove(&i64::from(episode_index))
                            .unwrap_or_default()
                    }),
                },
            )
        })
        .collect())
}

fn root_indices_for_columns(
    schema: &arrow_schema::Schema,
    wanted_column_names: &[&str],
) -> Vec<usize> {
    schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| wanted_column_names.contains(&field.name().as_str()))
        .map(|(position, _)| position)
        .collect()
}

fn episode_row_filter(
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    episode_indices: &[u32],
) -> RowFilter {
    let episode_filter_projection = ProjectionMask::columns(parquet_schema, [EPISODE_INDEX_COLUMN]);
    let selected_episode_indices: Arc<HashSet<i64>> = Arc::new(
        episode_indices
            .iter()
            .map(|&episode_index| i64::from(episode_index))
            .collect(),
    );
    let episode_filter = ArrowPredicateFn::new(episode_filter_projection, move |batch| {
        let episode_column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                ArrowError::CastError("episode_index predicate column is not int64".to_owned())
            })?;
        Ok(episode_column
            .iter()
            .map(|episode_index| {
                episode_index.map(|episode_index| selected_episode_indices.contains(&episode_index))
            })
            .collect::<BooleanArray>())
    });
    RowFilter::new(vec![Box::new(episode_filter)])
}

/// Every requested column with empty rows (no requested column exists in the
/// parquet).
fn empty_columns(
    requested_columns: &[&str],
    present_columns: &[&str],
    has_frame_index: bool,
    has_timestamp: bool,
) -> EpisodeVectorColumns {
    EpisodeVectorColumns {
        columns: requested_columns
            .iter()
            .map(|name| EpisodeVectorColumn {
                name: (*name).to_owned(),
                rows: Vec::new(),
            })
            .collect(),
        present_column_names: present_columns
            .iter()
            .map(|column_name| (*column_name).to_owned())
            .collect(),
        frame_indices: has_frame_index.then(Vec::new),
        timestamps: has_timestamp.then(Vec::new),
    }
}

enum TimestampColumn<'batch> {
    Float32(&'batch Float32Array),
    Float64(&'batch Float64Array),
}

fn timestamp_value(
    column: &ArrayRef,
    row: usize,
    data_parquet_path: &Path,
) -> Result<f64, LerobotError> {
    if let Some(values) = column.as_any().downcast_ref::<Float32Array>() {
        return Ok(if values.is_null(row) {
            f64::NAN
        } else {
            f64::from(values.value(row))
        });
    }
    if let Some(values) = column.as_any().downcast_ref::<Float64Array>() {
        return Ok(if values.is_null(row) {
            f64::NAN
        } else {
            values.value(row)
        });
    }
    Err(LerobotError::InconsistentMetadata {
        detail: format!(
            "{} projected timestamp is neither float32 nor float64",
            data_parquet_path.display()
        ),
    })
}

impl TimestampColumn<'_> {
    fn value(&self, row: usize) -> f64 {
        match self {
            Self::Float32(column) => {
                if column.is_null(row) {
                    f64::NAN
                } else {
                    f64::from(column.value(row))
                }
            }
            Self::Float64(column) => {
                if column.is_null(row) {
                    f64::NAN
                } else {
                    column.value(row)
                }
            }
        }
    }
}

fn require_timestamp_column<'batch>(
    batch: &'batch RecordBatch,
    path: &Path,
) -> Result<TimestampColumn<'batch>, LerobotError> {
    let column = batch.column_by_name(TIMESTAMP_COLUMN).ok_or_else(|| {
        LerobotError::InconsistentMetadata {
            detail: format!(
                "data parquet {} lacks a numeric {TIMESTAMP_COLUMN} column",
                path.display()
            ),
        }
    })?;
    if let Some(float32_column) = column.as_any().downcast_ref::<Float32Array>() {
        return Ok(TimestampColumn::Float32(float32_column));
    }
    if let Some(float64_column) = column.as_any().downcast_ref::<Float64Array>() {
        return Ok(TimestampColumn::Float64(float64_column));
    }
    Err(LerobotError::InconsistentMetadata {
        detail: format!(
            "data parquet {} has a non-numeric {TIMESTAMP_COLUMN} column",
            path.display()
        ),
    })
}

fn require_int64_column<'batch>(
    batch: &'batch RecordBatch,
    column_name: &str,
    path: &Path,
) -> Result<&'batch Int64Array, LerobotError> {
    batch
        .column_by_name(column_name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: format!(
                "data parquet {} lacks an int64 {column_name} column",
                path.display()
            ),
        })
}

/// Decodes one row of a `Float32` list column into a `Vec<f32>`. The parquet
/// stores the vector as a variable-size `list<element: float32>`, but arrow may
/// materialize it as a `List`, `LargeList`, or `FixedSizeList`; all three are
/// accepted. Null elements collapse to `0.0`.
fn decode_float32_list_row(
    column: &ArrayRef,
    row: usize,
    column_name: &str,
    path: &Path,
) -> Result<Vec<f32>, LerobotError> {
    let row_values: ArrayRef = if let Some(list) = column.as_any().downcast_ref::<ListArray>() {
        list.value(row)
    } else if let Some(list) = column.as_any().downcast_ref::<LargeListArray>() {
        list.value(row)
    } else if let Some(list) = column.as_any().downcast_ref::<FixedSizeListArray>() {
        list.value(row)
    } else {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!(
                "column {column_name:?} in {} is not a float32 list",
                path.display()
            ),
        });
    };

    let floats = row_values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: format!(
                "column {column_name:?} in {} does not hold float32 elements",
                path.display()
            ),
        })?;
    Ok((0..floats.len())
        .map(|element| {
            if floats.is_null(element) {
                0.0
            } else {
                floats.value(element)
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Float32Type;
    use arrow_array::{FixedSizeListArray, Float64Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::*;

    /// Writes a single data parquet holding every `(episode_index, per-frame
    /// 1-D action values)` concatenated in order — the v3 layout where one file
    /// packs many episodes' rows.
    fn write_shared_action_parquet(path: &Path, episodes: &[(u32, Vec<f32>)]) {
        let mut episode_index_column: Vec<i64> = Vec::new();
        let mut action_values: Vec<Option<Vec<Option<f32>>>> = Vec::new();
        for (episode_index, frames) in episodes {
            for &frame in frames {
                episode_index_column.push(i64::from(*episode_index));
                action_values.push(Some(vec![Some(frame)]));
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("episode_index", DataType::Int64, false),
            Field::new(
                "action",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 1),
                true,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(episode_index_column)),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(action_values, 1),
                ),
            ],
        )
        .expect("data batch");
        let file = File::create(path).expect("create parquet");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close parquet");
    }

    fn write_full_resolution_columns_parquet(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("episode_index", DataType::Int64, false),
            Field::new("frame_index", DataType::Int64, false),
            Field::new("timestamp", DataType::Float64, false),
            Field::new(
                "action",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 1),
                true,
            ),
        ]));
        let action_values = vec![
            Some(vec![Some(1.0)]),
            Some(vec![Some(2.0)]),
            Some(vec![Some(3.0)]),
        ];
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![2, 2, 2])),
                Arc::new(Int64Array::from(vec![10, 11, 13])),
                Arc::new(Float64Array::from(vec![0.0, 0.1, 0.3])),
                Arc::new(
                    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(action_values, 1),
                ),
            ],
        )
        .expect("data batch");
        let file = File::create(path).expect("create parquet");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("arrow writer");
        writer.write(&batch).expect("write batch");
        writer.close().expect("close parquet");
    }

    #[test]
    fn reads_many_episodes_from_one_shared_file_in_a_single_pass() {
        // Three episodes packed into ONE file: a single batch read must recover
        // each episode's own rows, in frame order, rather than one scan each.
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("file-000.parquet");
        write_shared_action_parquet(
            &data_path,
            &[
                (0, vec![0.0, 1.0]),
                (1, vec![10.0, 11.0, 12.0]),
                (2, vec![20.0]),
            ],
        );

        let columns_by_episode =
            read_episodes_vector_columns_from_file(&data_path, &[0, 1, 2], &["action"])
                .expect("batch read");

        assert_eq!(
            columns_by_episode.len(),
            3,
            "one entry per requested episode"
        );
        assert_eq!(
            columns_by_episode[&0u32]
                .column("action")
                .map(<[Vec<f32>]>::len),
            Some(2)
        );
        let episode_one = columns_by_episode[&1u32]
            .column("action")
            .expect("episode 1 action rows");
        assert_eq!(episode_one.len(), 3, "episode 1 keeps only its own rows");
        assert_eq!(episode_one[0], vec![10.0_f32]);
        assert_eq!(episode_one[2], vec![12.0_f32]);
        assert_eq!(
            columns_by_episode[&2u32]
                .column("action")
                .expect("episode 2 action rows")[0],
            vec![20.0_f32]
        );
    }

    #[test]
    fn read_plans_reject_row_claims_that_could_select_the_wrong_frames() {
        let mismatched_range = PlannedEpisodeDataRows::new(3, 3, Some(0..2))
            .expect_err("the claimed range is shorter than the episode");
        assert!(mismatched_range.detail().contains("expects 3 rows"));

        let first_episode =
            PlannedEpisodeDataRows::new(3, 3, Some(0..3)).expect("valid first episode");
        let overlapping_episode =
            PlannedEpisodeDataRows::new(7, 2, Some(2..4)).expect("valid range in isolation");
        let overlapping_plan = EpisodeDataFileReadPlan::new(
            PathBuf::from("packed.parquet"),
            vec![first_episode, overlapping_episode],
            Some(4),
        )
        .expect_err("overlapping episode ranges must not reach the parquet reader");
        assert!(overlapping_plan.detail().contains("overlaps"));
    }

    #[test]
    fn streaming_reader_prunes_to_exact_selected_episode_rows() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("file-000.parquet");
        write_shared_action_parquet(
            &data_path,
            &[
                (0, vec![0.0, 1.0]),
                (1, vec![10.0, 11.0, 12.0]),
                (2, vec![20.0]),
            ],
        );
        let plan = EpisodeDataFileReadPlan::new(
            data_path,
            vec![PlannedEpisodeDataRows::new(1, 3, Some(2..5)).expect("valid row plan")],
            Some(6),
        )
        .expect("valid packed-file plan");

        let mut reader = EpisodeVectorRowReader::open(&plan, &["action"]).expect("stream reader");
        let action_rows = reader
            .by_ref()
            .map(|row| row.expect("selected row").vectors[0].clone())
            .collect::<Vec<_>>();

        assert_eq!(
            action_rows,
            vec![vec![10.0_f32], vec![11.0_f32], vec![12.0_f32]]
        );
        assert_eq!(
            reader.report(),
            EpisodeVectorReadReport {
                selection_mode: EpisodeVectorSelectionMode::ExactRowRanges,
                source_row_count: 6,
                selected_row_count: 3,
                decoded_row_count: 3,
            }
        );
    }

    #[test]
    fn streaming_reader_falls_back_to_episode_predicate_without_exact_ranges() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("file-000.parquet");
        write_shared_action_parquet(
            &data_path,
            &[(0, vec![0.0, 1.0]), (1, vec![10.0]), (2, vec![20.0])],
        );
        let plan = EpisodeDataFileReadPlan::new(
            data_path,
            vec![PlannedEpisodeDataRows::new(2, 1, None).expect("valid fallback row plan")],
            None,
        )
        .expect("valid predicate-fallback plan");

        let mut reader = EpisodeVectorRowReader::open(&plan, &["action"]).expect("stream reader");
        let rows = reader
            .by_ref()
            .collect::<Result<Vec<_>, _>>()
            .expect("predicate rows");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].vectors[0], vec![20.0_f32]);
        assert_eq!(
            reader.report().selection_mode,
            EpisodeVectorSelectionMode::EpisodeIndexPredicate
        );
    }

    #[test]
    fn full_resolution_read_includes_raw_timestamps_and_frame_indices() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("file-000.parquet");
        write_full_resolution_columns_parquet(&data_path);

        let columns_by_episode =
            read_episodes_vector_columns_from_file(&data_path, &[2], &["action"])
                .expect("full-resolution read");
        let columns = &columns_by_episode[&2];

        assert!(columns.column_is_present("action"));
        assert_eq!(columns.frame_indices(), Some([10, 11, 13].as_slice()));
        assert_eq!(columns.timestamps(), Some([0.0, 0.1, 0.3].as_slice()));
    }

    #[test]
    fn full_resolution_read_preserves_missing_bookkeeping_columns() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("file-000.parquet");
        write_shared_action_parquet(&data_path, &[(0, vec![1.0])]);

        let columns_by_episode =
            read_episodes_vector_columns_from_file(&data_path, &[0], &["action"])
                .expect("vector-only read");
        let columns = &columns_by_episode[&0];

        assert!(columns.column_is_present("action"));
        assert!(columns.frame_indices().is_none());
        assert!(columns.timestamps().is_none());
    }

    #[test]
    fn a_requested_episode_absent_from_the_file_reads_back_empty() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let data_path = temp_dir.path().join("file-000.parquet");
        write_shared_action_parquet(&data_path, &[(0, vec![1.0, 2.0])]);

        // Episode 7 has no rows in the file: it still gets an entry with empty
        // rows so callers can align positionally without a missing-key panic.
        let columns_by_episode =
            read_episodes_vector_columns_from_file(&data_path, &[0, 7], &["action"])
                .expect("batch read");
        assert_eq!(
            columns_by_episode[&0u32]
                .column("action")
                .map(<[Vec<f32>]>::len),
            Some(2)
        );
        assert_eq!(
            columns_by_episode[&7u32].column("action"),
            Some([].as_slice()),
            "an episode absent from the file reads back empty, not missing"
        );
    }

    #[test]
    fn column_lookup_returns_requested_rows_and_none_for_unrequested() {
        let columns = EpisodeVectorColumns {
            columns: vec![
                EpisodeVectorColumn {
                    name: "action".to_owned(),
                    rows: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                },
                EpisodeVectorColumn {
                    name: "observation.state".to_owned(),
                    rows: Vec::new(),
                },
            ],
            present_column_names: HashSet::from(["action".to_owned()]),
            frame_indices: None,
            timestamps: None,
        };
        assert_eq!(columns.column("action").map(<[Vec<f32>]>::len), Some(2));
        // A requested-but-absent column reads back as an empty slice, not None.
        assert_eq!(
            columns.column("observation.state"),
            Some([].as_slice()),
            "a requested column absent from the parquet is an empty slice"
        );
        // A column that was never requested is absent from the result entirely.
        assert!(columns.column("timestamp").is_none());
    }

    #[test]
    fn empty_columns_preserves_request_order_with_no_rows() {
        let columns = empty_columns(&["observation.state", "action"], &[], false, false);
        let names: Vec<&str> = columns
            .columns()
            .iter()
            .map(|column| column.name.as_str())
            .collect();
        assert_eq!(names, ["observation.state", "action"]);
        assert!(
            columns
                .columns()
                .iter()
                .all(|column| column.rows.is_empty())
        );
    }
}
