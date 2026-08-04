use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, mpsc};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;

use arrow_array::{Array, RecordBatch, StringArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::arrow_columns::{f64_row, require_list_column, row_is_null, u32_row, u64_row};
use crate::{
    DatasetCachePolicy, DatasetFileMaterialization, DatasetInfo, DatasetRelativePath,
    DatasetSource, FramesPerSecond, LerobotError, RemoteDatasetSource, TimestampRange,
    render_path_template,
};

const DATASET_OPEN_PROGRESS_OBSERVATION_INTERVAL: std::time::Duration =
    std::time::Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectStoreCacheAccess {
    Private,
    SharedReadOnly,
}

static OBJECT_STORE_CACHE_CLEANUP_SENDER: OnceLock<Option<mpsc::Sender<PathBuf>>> = OnceLock::new();

/// Shared lifetime owner for one materialized object-store cache. The tempdir
/// is persisted into a plain path at construction so its destructor can never
/// perform recursive filesystem work on whichever thread drops the last
/// [`Dataset`].
#[derive(Debug)]
struct ObjectStoreCache {
    path: PathBuf,
    cleanup_sender: Option<mpsc::Sender<PathBuf>>,
}

impl ObjectStoreCache {
    fn from_temp_directory(temp_directory: tempfile::TempDir) -> Self {
        // Initialize the worker while cache creation is already running on a
        // blocking thread. Drop only performs an unbounded-channel send.
        let cleanup_sender = object_store_cache_cleanup_sender();
        Self {
            path: temp_directory.keep(),
            cleanup_sender,
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ObjectStoreCache {
    fn drop(&mut self) {
        let cache_path = std::mem::take(&mut self.path);
        let Some(cleanup_sender) = &self.cleanup_sender else {
            tracing::warn!(
                cache_path = %cache_path.display(),
                "leaking object-store dataset cache because its cleanup worker is unavailable"
            );
            return;
        };
        if let Err(schedule_error) = cleanup_sender.send(cache_path) {
            tracing::warn!(
                cache_path = %schedule_error.0.display(),
                "leaking object-store dataset cache because cleanup scheduling failed"
            );
        }
    }
}

fn object_store_cache_cleanup_sender() -> Option<mpsc::Sender<PathBuf>> {
    OBJECT_STORE_CACHE_CLEANUP_SENDER
        .get_or_init(|| {
            let (cleanup_sender, cleanup_receiver) = mpsc::channel::<PathBuf>();
            match std::thread::Builder::new()
                .name("lerobot-dataset-cache-cleanup".to_owned())
                .spawn(move || {
                    while let Ok(cache_path) = cleanup_receiver.recv() {
                        if let Err(cleanup_error) = fs::remove_dir_all(&cache_path)
                            && cleanup_error.kind() != io::ErrorKind::NotFound
                        {
                            tracing::warn!(
                                cache_path = %cache_path.display(),
                                error = %cleanup_error,
                                "failed to remove object-store dataset cache"
                            );
                        }
                    }
                }) {
                Ok(_cleanup_worker) => Some(cleanup_sender),
                Err(spawn_error) => {
                    tracing::warn!(
                        error = %spawn_error,
                        "failed to start object-store dataset cache cleanup worker"
                    );
                    None
                }
            }
        })
        .clone()
}

/// A measurable edge while opening dataset metadata. Callers may use these
/// events for liveness without turning an unconditional timer into fake
/// progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetOpenProgress {
    /// One more metadata key arrived from the object-store listing stream.
    MetadataObjectListingAdvanced {
        /// Number of metadata objects discovered so far.
        objects_discovered: u64,
    },
    /// Object-store metadata listing completed and established the work total.
    MetadataObjectsListed {
        /// Total number of metadata objects that will be materialized.
        total_objects: u64,
    },
    /// Bytes of the current metadata object visibly advanced in the cache.
    MetadataObjectDownloadAdvanced {
        /// One-based position of the object currently being materialized.
        object_ordinal: u64,
        /// Total number of metadata objects in the open operation.
        total_objects: u64,
        /// Bytes materialized for the current object so far.
        bytes_downloaded: u64,
    },
    /// One more complete metadata object was atomically materialized.
    MetadataObjectDownloaded {
        /// Number of metadata objects completely materialized so far.
        completed_objects: u64,
        /// Total number of metadata objects in the open operation.
        total_objects: u64,
    },
    /// Another parquet record batch of episode metadata was decoded.
    EpisodeMetadataRowsParsed {
        /// Cumulative episode-metadata rows decoded so far.
        rows_parsed: u64,
    },
    /// Another record batch owned by a format extension was decoded.
    ExtensionRowsParsed {
        /// Stable extension identifier suitable for logs and progress events.
        extension_name: &'static str,
        /// Cumulative rows decoded for that extension.
        rows_parsed: u64,
    },
}

/// Callback invoked when opening a dataset makes measurable progress.
pub type DatasetOpenProgressReporter<'reporter> =
    dyn Fn(DatasetOpenProgress) + Send + Sync + 'reporter;

/// One episode's slice of a concatenated video file, in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoSegment {
    /// Absolute path of the video file, resolved against the dataset root.
    file: PathBuf,
    /// Chunk component used to render `file` from the dataset's path template.
    chunk_index: u32,
    /// Packed file component used to render `file` from the path template.
    file_index: u32,
    timestamp_range: TimestampRange,
}

impl VideoSegment {
    /// Creates a segment whose timestamp range has already been validated.
    #[must_use]
    pub fn new(
        file: PathBuf,
        chunk_index: u32,
        file_index: u32,
        timestamp_range: TimestampRange,
    ) -> Self {
        Self {
            file,
            chunk_index,
            file_index,
            timestamp_range,
        }
    }

    /// Local source path or remote-cache destination for the packed video.
    #[must_use]
    pub fn file(&self) -> &Path {
        &self.file
    }

    /// Chunk index used by the source video-path template.
    #[must_use]
    pub const fn chunk_index(&self) -> u32 {
        self.chunk_index
    }

    /// Packed-file index used by the source video-path template.
    #[must_use]
    pub const fn file_index(&self) -> u32 {
        self.file_index
    }

    /// Validated inclusive-start, exclusive-end range within the packed file.
    #[must_use]
    pub const fn timestamp_range(&self) -> TimestampRange {
        self.timestamp_range
    }
}

/// Per-episode metadata read from `meta/episodes/chunk-*/file-*.parquet`.
#[derive(Debug, Clone)]
// `episode_index` is the exact format term; shortening it to `index` would be
// ambiguous beside the dataset row indices carried by the same value.
#[allow(clippy::struct_field_names)]
pub struct Episode {
    episode_index: u32,
    length_frames: u64,
    tasks: Vec<String>,
    /// Video segment per camera key this episode HAS footage for — classified
    /// RGB keys and encoded depth keys alike.
    ///
    /// A declared camera may be MISSING here. `LeRobot` v3 gives every camera a
    /// `videos/{key}/*` column group on every episode row, so nulling that
    /// whole group is the only way a source can say "this episode has no such
    /// camera" — a partially-instrumented rig, a sensor that dropped out
    /// mid-collection, a conversion that padded a stream it never recorded.
    /// See [`video_segment_from_row`].
    ///
    /// Consumers must therefore not assume this map is keyed by the dataset's
    /// full camera set. Callers that tolerate sparse cameras may skip an absent
    /// (episode, camera) pair. An export that explicitly retains that camera
    /// must instead fail, because silently omitting it would produce a subset
    /// whose metadata lies about its own contents.
    video_segments: BTreeMap<String, VideoSegment>,
    /// Inclusive global frame index of the episode's first frame.
    dataset_from_index: u64,
    /// Exclusive global frame index just past the episode's last frame.
    dataset_to_index: u64,
    data_chunk_index: u32,
    data_file_index: u32,
}

impl Episode {
    /// Stable episode index declared by the source dataset.
    #[must_use]
    pub const fn episode_index(&self) -> u32 {
        self.episode_index
    }

    /// Number of frame rows in this episode.
    #[must_use]
    pub const fn length_frames(&self) -> u64 {
        self.length_frames
    }

    /// Task labels attached to this episode.
    #[must_use]
    pub fn tasks(&self) -> &[String] {
        &self.tasks
    }

    /// Available video segments keyed by feature name.
    #[must_use]
    pub fn video_segments(&self) -> &BTreeMap<String, VideoSegment> {
        &self.video_segments
    }

    /// Inclusive first row in the dataset-global frame table.
    #[must_use]
    pub const fn dataset_from_index(&self) -> u64 {
        self.dataset_from_index
    }

    /// Exclusive end row in the dataset-global frame table.
    #[must_use]
    pub const fn dataset_to_index(&self) -> u64 {
        self.dataset_to_index
    }

    /// Chunk index of the packed data file containing this episode.
    #[must_use]
    pub const fn data_chunk_index(&self) -> u32 {
        self.data_chunk_index
    }

    /// File index of the packed data file containing this episode.
    #[must_use]
    pub const fn data_file_index(&self) -> u32 {
        self.data_file_index
    }

    #[must_use]
    /// Duration derived from this episode's frame count and `frames_per_second`.
    pub fn duration_secs(&self, frames_per_second: FramesPerSecond) -> f64 {
        frames_per_second.duration_for_frames(self.length_frames)
    }
}

/// A `LeRobot` v3 dataset opened for reading.
///
/// The dataset's raw files live at a [`DatasetSource`] — a local directory or
/// a remote provider. Reading always happens against LOCAL paths under
/// [`Dataset::root`]: for a local source that is the canonicalized dataset
/// directory itself; for a remote source it is a temp cache into which
/// metadata is materialized eagerly and video/data files are downloaded on
/// demand (see [`Dataset::episode_data_parquet_path`] and
/// [`Dataset::ensure_segment_local`]).
#[derive(Debug)]
pub struct Dataset {
    source: DatasetSource,
    /// The LOCAL root every returned file path resolves under: the canonical
    /// dataset directory (local source) or the temp materialization cache
    /// (remote source).
    local_root: PathBuf,
    /// Keeps the temp cache alive for a remote source; `None` for a local
    /// source, whose files are read in place.
    _local_cache: Option<Arc<ObjectStoreCache>>,
    object_store_cache_access: ObjectStoreCacheAccess,
    info: DatasetInfo,
    episodes: Vec<Episode>,
    video_keys: Vec<String>,
}

impl Dataset {
    /// Opens the LOCAL dataset rooted at `root` by parsing `meta/info.json` and
    /// all episode metadata parquet files. This is the unchanged filesystem
    /// fast path; for a remote dataset use [`Dataset::open_source`].
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Io`] when the root or required files are
    /// missing, [`LerobotError::UnsupportedVersion`] for non-v3 datasets, and
    /// [`LerobotError::Json`] / [`LerobotError::Parquet`] /
    /// [`LerobotError::MissingColumn`] / [`LerobotError::InconsistentMetadata`]
    /// when metadata is malformed or internally inconsistent.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, LerobotError> {
        Self::open_local(root.as_ref())
    }

    /// Opens a local dataset synchronously while reporting concrete parquet
    /// parse edges. Async callers must invoke this from a blocking task; most
    /// should prefer [`Self::open_source_with_progress`], which performs that
    /// handoff itself.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::open`].
    pub fn open_with_progress(
        root: impl AsRef<Path>,
        report_progress: &(dyn Fn(DatasetOpenProgress) + Send + Sync),
    ) -> Result<Self, LerobotError> {
        Self::open_local_with_progress(root.as_ref(), report_progress)
    }

    /// The shared local-directory open path: canonicalize, then parse metadata
    /// in place. Never touches the network, so it stays synchronous.
    fn open_local(root: &Path) -> Result<Self, LerobotError> {
        Self::open_local_with_progress(root, &|_| {})
    }

    fn open_local_with_progress(
        root: &Path,
        progress_reporter: &DatasetOpenProgressReporter<'_>,
    ) -> Result<Self, LerobotError> {
        let canonical = fs::canonicalize(root).map_err(|io_error| LerobotError::Io {
            path: root.to_path_buf(),
            source: io_error,
        })?;
        Self::from_local_metadata(
            DatasetSource::Local(canonical.clone()),
            canonical,
            None,
            ObjectStoreCacheAccess::Private,
            progress_reporter,
        )
    }

    /// Opens the dataset at `source` — a local directory (identical to
    /// [`Dataset::open`]) or a remote location.
    ///
    /// For a remote source the entire `meta/` subtree is downloaded into
    /// a temp cache and parsed there, so all metadata (info, episodes, tasks,
    /// stats) reads exactly as a local dataset; the larger video and data files
    /// are fetched when a caller resolves them through the async ensure/prefetch
    /// methods. Downloads are awaited directly on the caller's runtime (there is
    /// no sync↔async bridge anywhere in this crate), and the CPU-bound metadata
    /// parse runs on a blocking thread, so this is safe to call from a Tokio
    /// async context.
    ///
    /// # Errors
    ///
    /// As [`Dataset::open`], plus [`LerobotError::Source`] when object metadata
    /// cannot be listed or downloaded.
    pub async fn open_source(source: &DatasetSource) -> Result<Self, LerobotError> {
        Self::open_source_with_progress(source, |_| {}).await
    }

    /// Opens `source` while reporting only concrete metadata download and
    /// parse progress. The callback may run on either an async runtime worker
    /// or a blocking metadata-parser thread and therefore must be thread-safe.
    ///
    /// # Errors
    ///
    /// As [`Self::open_source`].
    pub async fn open_source_with_progress<ReportProgress>(
        source: &DatasetSource,
        report_progress: ReportProgress,
    ) -> Result<Self, LerobotError>
    where
        ReportProgress: Fn(DatasetOpenProgress) + Send + Sync + 'static,
    {
        let progress_reporter: Arc<DatasetOpenProgressReporter<'static>> =
            Arc::new(report_progress);
        match source {
            DatasetSource::Local(path) => {
                let owned_path = path.clone();
                let task_path = owned_path.clone();
                let task_progress_reporter = Arc::clone(&progress_reporter);
                // Local opens parse the same potentially large parquet metadata
                // tree as object-store opens, so they must not occupy a runtime
                // worker while doing filesystem and CPU-bound work.
                tokio::task::spawn_blocking(move || {
                    Self::open_local_with_progress(&owned_path, task_progress_reporter.as_ref())
                })
                .await
                .map_err(|join_error| LerobotError::Io {
                    path: task_path,
                    source: std::io::Error::other(format!(
                        "local dataset metadata task failed: {join_error}"
                    )),
                })?
            }
            DatasetSource::Remote(remote_source) => {
                let (cache, cache_access) =
                    create_object_store_cache_async(remote_source.cache_policy()).await?;
                materialize_metadata_tree(
                    remote_source,
                    cache.path(),
                    cache_access,
                    progress_reporter.as_ref(),
                )
                .await?;
                let local_root = cache.path().to_path_buf();
                let task_local_root = local_root.clone();
                let owned_source = source.clone();
                let cache = Arc::new(cache);
                let task_progress_reporter = Arc::clone(&progress_reporter);
                // Parsing 30k+ episode metadata rows is CPU-bound parquet work;
                // keep it off the runtime workers.
                tokio::task::spawn_blocking(move || {
                    Self::from_local_metadata(
                        owned_source,
                        local_root,
                        Some(cache),
                        cache_access,
                        task_progress_reporter.as_ref(),
                    )
                })
                .await
                .map_err(|join_error| LerobotError::Io {
                    path: task_local_root,
                    source: std::io::Error::other(format!(
                        "object-store dataset metadata task failed: {join_error}"
                    )),
                })?
            }
        }
    }

    /// Parses the dataset metadata already present under `local_root/meta` and
    /// builds the handle. Shared by both source variants (single source of
    /// truth for the parse).
    fn from_local_metadata(
        source: DatasetSource,
        local_root: PathBuf,
        local_cache: Option<Arc<ObjectStoreCache>>,
        object_store_cache_access: ObjectStoreCacheAccess,
        progress_reporter: &DatasetOpenProgressReporter<'_>,
    ) -> Result<Self, LerobotError> {
        let info_path = local_root.join("meta").join("info.json");
        let info_json_text =
            fs::read_to_string(&info_path).map_err(|io_error| LerobotError::Io {
                path: info_path,
                source: io_error,
            })?;
        let info = DatasetInfo::parse_json(&info_json_text)?;
        // The stored `video_keys` field stays RGB-only: it is the "what may
        // be embedded/indexed" contract every camera validator leans on.
        // Episode SEGMENTS are read for RGB and depth alike so planners,
        // exporters, and previews can carry depth views through.
        let video_keys = info.rgb_video_keys();
        let mut segment_read_keys = video_keys.clone();
        segment_read_keys.extend(info.depth_video_keys());

        let episodes =
            read_all_episode_metadata(&local_root, &info, &segment_read_keys, progress_reporter)?;
        validate_dataset_invariants(&info, &episodes)?;

        Ok(Self {
            source,
            local_root,
            _local_cache: local_cache,
            object_store_cache_access,
            info,
            episodes,
            video_keys,
        })
    }

    /// The LOCAL root every returned file path resolves under: the canonicalized
    /// dataset directory for a local source, or the temp materialization cache
    /// for a remote source. Use [`Dataset::source`] /
    /// [`Dataset::root_identifier`] for the dataset's stable identity.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.local_root
    }

    /// Where this dataset's raw files live (local path or remote provider).
    #[must_use]
    pub fn source(&self) -> &DatasetSource {
        &self.source
    }

    /// The stable identifier under which this dataset is recorded and reopened:
    /// the canonical local path (local source) or remote identifier. This is
    /// what the registry and index manifest store; passing it back through
    /// a deployment adapter can use to reopen the same dataset.
    #[must_use]
    pub fn root_identifier(&self) -> PathBuf {
        match &self.source {
            DatasetSource::Local(_) => self.local_root.clone(),
            DatasetSource::Remote(remote_source) => PathBuf::from(remote_source.identifier()),
        }
    }

    /// Ensures the video file backing `segment` is available locally and returns
    /// its local path. A no-op returning `segment.file` for a local source; for
    /// a remote source it awaits the download of the video chunk into the
    /// temp cache on first use (idempotent). Callers that hand the path to
    /// ffmpeg must await this first.
    ///
    /// Concurrent callers targeting the same absent destination must use a
    /// per-path single-flight coordinator; distinct destinations are safe to
    /// download concurrently.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Source`] / [`LerobotError::Io`] when the video
    /// object cannot be read or cached.
    pub async fn ensure_segment_local(
        &self,
        segment: &VideoSegment,
    ) -> Result<PathBuf, LerobotError> {
        let relative_key = self.video_segment_relative_key(segment)?;
        self.ensure_relative_local(&relative_key).await
    }

    /// LOCAL path of the video file backing `segment` WITHOUT downloading:
    /// `segment.file` for a local source; for a remote source the cached
    /// path, which must already have been materialized by
    /// [`Self::ensure_segment_local`] / [`Self::prefetch_video_segments`].
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::NotMaterialized`] when a remotely backed
    /// video has not been downloaded yet, and [`LerobotError::Source`] if the
    /// segment does not belong to this dataset.
    pub fn local_segment_path(&self, segment: &VideoSegment) -> Result<PathBuf, LerobotError> {
        let relative_key = self.video_segment_relative_key(segment)?;
        self.require_relative_local(&relative_key)
    }

    /// Dataset-relative object key of the packed video containing `segment`.
    ///
    /// The returned key always uses forward slashes, so it can be recorded in
    /// a portable acquisition/copy manifest even when the source is local.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Source`] if the segment does not belong to this
    /// dataset.
    pub fn video_segment_relative_key(
        &self,
        segment: &VideoSegment,
    ) -> Result<String, LerobotError> {
        self.video_file_relative_key(segment.file())
    }

    /// Dataset-relative object key of a packed video FILE.
    ///
    /// A segment resolves to bytes through its `file` alone — the rendered
    /// path components are template inputs, not addressing — so callers that
    /// hold only a path (the source-contract materializer) use this directly.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Source`] if the path is not under this
    /// dataset's root.
    pub fn video_file_relative_key(&self, file: &Path) -> Result<String, LerobotError> {
        let relative_path =
            file.strip_prefix(&self.local_root)
                .map_err(|_| LerobotError::Source {
                    context: format!("resolving video file {}", file.display()),
                    detail: "video path is not under the dataset root cache".to_owned(),
                })?;
        Ok(to_forward_slash_key(relative_path))
    }

    /// Resolves a packed video FILE to a readable local path, downloading it
    /// for a remote source. The path-addressed twin of
    /// [`Self::ensure_segment_local`].
    ///
    /// # Errors
    ///
    /// Returns an error when the file is not under this dataset's root or the
    /// download fails.
    pub async fn ensure_local_video_file(&self, file: &Path) -> Result<PathBuf, LerobotError> {
        let relative_key = self.video_file_relative_key(file)?;
        self.ensure_relative_local(&relative_key).await
    }

    /// Whether media files under [`Self::root`] are a DOWNLOADED CACHE of an
    /// remote source (safe to delete once consumed and re-downloadable
    /// on demand) rather than the local source itself (never delete).
    #[must_use]
    pub fn is_remote_cached(&self) -> bool {
        matches!(self.source, DatasetSource::Remote(_))
    }

    #[must_use]
    /// Parsed dataset-level metadata from `meta/info.json`.
    pub fn info(&self) -> &DatasetInfo {
        &self.info
    }

    /// All episodes, sorted by `episode_index`.
    #[must_use]
    pub fn episodes(&self) -> &[Episode] {
        &self.episodes
    }

    #[must_use]
    /// Returns the episode with `episode_index`, if it exists.
    pub fn episode(&self, episode_index: u32) -> Option<&Episode> {
        self.episodes
            .binary_search_by_key(&episode_index, |episode| episode.episode_index)
            .ok()
            .map(|position| &self.episodes[position])
    }

    /// LOCAL path of the data parquet file that stores `episode_index`'s
    /// per-frame rows, downloading it from a remote source on first use.
    ///
    /// A v3 data file concatenates MANY episodes, so grouping episodes by this
    /// path lets a batch read scan each file exactly once (see
    /// [`crate::read_episodes_vector_columns_from_file`]). Every data-parquet
    /// reader in this crate resolves the file through THIS method (or the
    /// prefetch + [`Self::local_episode_data_parquet_path`] pair), so the
    /// awaited download is the single seam that makes remote datasets
    /// readable without touching the readers.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::EpisodeNotFound`] for an unknown episode index,
    /// [`LerobotError::InconsistentMetadata`] when the data-path template
    /// cannot be rendered, and [`LerobotError::Source`] / [`LerobotError::Io`]
    /// when the data object cannot be downloaded (remote source).
    pub async fn episode_data_parquet_path(
        &self,
        episode_index: u32,
    ) -> Result<PathBuf, LerobotError> {
        let relative_data_path = self.episode_data_relative_key(episode_index)?;
        self.ensure_relative_local(&relative_data_path).await
    }

    /// LOCAL path of the data parquet backing `episode_index` WITHOUT
    /// downloading; for a remote source it must already have been
    /// materialized by [`Self::prefetch_episode_data`] /
    /// [`Self::episode_data_parquet_path`].
    ///
    /// # Errors
    ///
    /// As [`Self::episode_data_parquet_path`], except a missing remote
    /// download surfaces as [`LerobotError::NotMaterialized`].
    pub fn local_episode_data_parquet_path(
        &self,
        episode_index: u32,
    ) -> Result<PathBuf, LerobotError> {
        let relative_data_path = self.episode_data_relative_key(episode_index)?;
        self.require_relative_local(&relative_data_path)
    }

    /// Dataset-relative object key of the packed data parquet containing an
    /// episode. Planning callers can use this without materializing the data
    /// object from remote storage.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::EpisodeNotFound`] for an unknown episode or
    /// [`LerobotError::InconsistentMetadata`] for an invalid path template.
    pub fn episode_data_relative_key(&self, episode_index: u32) -> Result<String, LerobotError> {
        let episode = self
            .episode(episode_index)
            .ok_or(LerobotError::EpisodeNotFound {
                index: episode_index,
            })?;
        let relative_data_path = render_path_template(
            self.info.data_path_template(),
            None,
            episode.data_chunk_index,
            episode.data_file_index,
        )?;
        Ok(relative_data_path.into_string())
    }

    /// Resolves a forward-slash `relative_key` to a LOCAL path under
    /// [`Self::root`] WITHOUT downloading, failing loud when an
    /// remotely backed file has not been materialized yet. This is what the
    /// synchronous read paths (the export pipeline) resolve through, so a
    /// missed prefetch is an immediate, explicit error instead of a hidden
    /// network call.
    fn require_relative_local(&self, relative_key: &str) -> Result<PathBuf, LerobotError> {
        let local_path = safe_dataset_relative_path(&self.local_root, relative_key)?;
        if matches!(self.source, DatasetSource::Remote(_)) && !local_path.exists() {
            return Err(LerobotError::NotMaterialized {
                relative_key: relative_key.to_owned(),
            });
        }
        Ok(local_path)
    }

    /// Materializes the packed data parquet files backing `episode_indices`
    /// BEFORE any synchronous read touches them, so hot per-episode loops
    /// batch processing downloads with bounded concurrency up
    /// front instead of serially at first use. Episodes sharing a packed file
    /// are deduplicated. A local dataset (or an already-warm cache) is a cheap
    /// no-op. Episodes absent from the raw dataset are skipped, mirroring the
    /// readers' drift tolerance.
    ///
    /// Returns the number of distinct data files ensured local.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::InconsistentMetadata`] for an invalid path
    /// template or [`LerobotError::Source`]/[`LerobotError::Io`] when a
    /// download fails.
    pub async fn prefetch_episode_data(
        &self,
        episode_indices: &[u32],
    ) -> Result<u64, LerobotError> {
        if !matches!(self.source, DatasetSource::Remote(_)) {
            return Ok(0);
        }
        let mut relative_keys = std::collections::BTreeSet::new();
        for &episode_index in episode_indices {
            match self.episode_data_relative_key(episode_index) {
                Ok(relative_key) => {
                    relative_keys.insert(relative_key);
                }
                Err(LerobotError::EpisodeNotFound { .. }) => {}
                Err(other_error) => return Err(other_error),
            }
        }
        self.prefetch_relative_keys(relative_keys).await
    }

    /// Materializes the packed video files backing `episode_indices` ×
    /// `video_keys` BEFORE any synchronous read touches them (the export
    /// pipeline resolves videos through [`Self::local_segment_path`]). Packed
    /// files shared between episodes are deduplicated; a local dataset is a
    /// cheap no-op. Unknown episode indices are skipped, mirroring
    /// [`Self::prefetch_episode_data`]; an episode MISSING one of the requested
    /// cameras is skipped too, deferring the inconsistency to the reader that
    /// actually needs the segment.
    ///
    /// Returns the number of distinct video files ensured local.
    ///
    /// # Errors
    ///
    /// Returns [`LerobotError::Source`] for a segment outside the dataset root
    /// or [`LerobotError::Io`] when a download fails.
    pub async fn prefetch_video_segments(
        &self,
        episode_indices: &[u32],
        video_keys: &[String],
    ) -> Result<u64, LerobotError> {
        if !matches!(self.source, DatasetSource::Remote(_)) {
            return Ok(0);
        }
        let mut relative_keys = std::collections::BTreeSet::new();
        for &episode_index in episode_indices {
            let Some(episode) = self.episode(episode_index) else {
                continue;
            };
            for video_key in video_keys {
                let Some(segment) = episode.video_segments.get(video_key) else {
                    continue;
                };
                relative_keys.insert(self.video_segment_relative_key(segment)?);
            }
        }
        self.prefetch_relative_keys(relative_keys).await
    }

    async fn prefetch_relative_keys(
        &self,
        relative_keys: std::collections::BTreeSet<String>,
    ) -> Result<u64, LerobotError> {
        use futures::StreamExt as _;

        const PREFETCH_CONCURRENCY: usize = 8;
        let ensured = futures::stream::iter(relative_keys)
            .map(|relative_key| async move { self.ensure_relative_local(&relative_key).await })
            .buffer_unordered(PREFETCH_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
        let mut ensured_count: u64 = 0;
        for ensure_outcome in ensured {
            ensure_outcome?;
            ensured_count += 1;
        }
        Ok(ensured_count)
    }

    /// Resolves a forward-slash `relative_key` (relative to the dataset root) to
    /// a LOCAL path under [`Self::root`], awaiting its download from an
    /// remote source when absent. A pure `local_root.join` for a local
    /// source.
    async fn ensure_relative_local(&self, relative_key: &str) -> Result<PathBuf, LerobotError> {
        let local_path = safe_dataset_relative_path(&self.local_root, relative_key)?;
        if let DatasetSource::Remote(remote_source) = &self.source {
            let local_object_exists =
                tokio::fs::try_exists(&local_path)
                    .await
                    .map_err(|source| LerobotError::Io {
                        path: local_path.clone(),
                        source,
                    })?;
            if !local_object_exists {
                download_object_to(
                    remote_source,
                    relative_key,
                    &local_path,
                    &self.local_root,
                    self.object_store_cache_access,
                )
                .await?;
            }
        }
        Ok(local_path)
    }

    /// Classified RGB camera keys in `meta/info.json` document order.
    #[must_use]
    pub fn video_keys(&self) -> &[String] {
        &self.video_keys
    }

    /// The dataset's classified DEPTH camera keys. Their segments are
    /// present in [`Episode::video_segments`] for planning/export/preview
    /// use, but they are never embeddable ([`Self::video_keys`] stays RGB).
    #[must_use]
    pub fn depth_video_keys(&self) -> Vec<String> {
        self.info.depth_video_keys()
    }

    /// Every classified camera key, RGB then depth, in document order.
    #[must_use]
    pub fn all_video_keys(&self) -> Vec<String> {
        let mut all_keys = self.video_keys.clone();
        all_keys.extend(self.info.depth_video_keys());
        all_keys
    }

    #[must_use]
    /// Validated frame rate declared by `meta/info.json`.
    pub const fn frames_per_second(&self) -> FramesPerSecond {
        self.info.frames_per_second()
    }
}

/// Creates the per-dataset object-store materialization cache selected by the
/// source adapter. Generic providers use a private system-temp directory;
/// deployment adapters can opt into a shared read-only parent.
async fn create_object_store_cache_async(
    cache_policy: &DatasetCachePolicy,
) -> Result<(ObjectStoreCache, ObjectStoreCacheAccess), LerobotError> {
    let owned_cache_policy = cache_policy.clone();
    let task_error_path = match &owned_cache_policy {
        DatasetCachePolicy::PrivateTemporaryDirectory => std::env::temp_dir(),
        DatasetCachePolicy::SharedReadOnlyDirectory {
            parent_directory, ..
        } => parent_directory.clone(),
    };
    tokio::task::spawn_blocking(move || create_object_store_cache(owned_cache_policy))
        .await
        .map_err(|join_error| LerobotError::Io {
            path: task_error_path,
            source: io::Error::other(format!(
                "object-store dataset cache creation task failed: {join_error}"
            )),
        })?
}

fn create_object_store_cache(
    cache_policy: DatasetCachePolicy,
) -> Result<(ObjectStoreCache, ObjectStoreCacheAccess), LerobotError> {
    match cache_policy {
        DatasetCachePolicy::PrivateTemporaryDirectory => tempfile::tempdir()
            .map(|cache| {
                (
                    ObjectStoreCache::from_temp_directory(cache),
                    ObjectStoreCacheAccess::Private,
                )
            })
            .map_err(|source| LerobotError::Io {
                path: std::env::temp_dir(),
                source,
            }),
        DatasetCachePolicy::SharedReadOnlyDirectory {
            parent_directory,
            temporary_directory_prefix,
            configuration_name,
        } => create_shared_object_store_cache(
            &parent_directory,
            &temporary_directory_prefix,
            &configuration_name,
        ),
    }
}

fn create_shared_object_store_cache(
    shared_cache_parent: &Path,
    temporary_directory_prefix: &str,
    configuration_name: &str,
) -> Result<(ObjectStoreCache, ObjectStoreCacheAccess), LerobotError> {
    if shared_cache_parent.as_os_str().is_empty() || !shared_cache_parent.is_absolute() {
        return Err(LerobotError::Io {
            path: shared_cache_parent.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{configuration_name} must be an absolute path"),
            ),
        });
    }

    fs::create_dir_all(shared_cache_parent).map_err(|source| LerobotError::Io {
        path: shared_cache_parent.to_path_buf(),
        source,
    })?;
    set_shared_directory_permissions(shared_cache_parent).map_err(|source| LerobotError::Io {
        path: shared_cache_parent.to_path_buf(),
        source,
    })?;

    let cache = tempfile::Builder::new()
        .prefix(temporary_directory_prefix)
        .tempdir_in(shared_cache_parent)
        .map_err(|source| LerobotError::Io {
            path: shared_cache_parent.to_path_buf(),
            source,
        })?;
    set_shared_directory_permissions(cache.path()).map_err(|source| LerobotError::Io {
        path: cache.path().to_path_buf(),
        source,
    })?;
    Ok((
        ObjectStoreCache::from_temp_directory(cache),
        ObjectStoreCacheAccess::SharedReadOnly,
    ))
}

/// Downloads every object under the source's `meta/` prefix into
/// `cache_root/meta/...`, so the whole metadata subtree (info, episodes, tasks,
/// stats) parses locally exactly as an on-disk dataset. Video and data files are
/// deliberately left for lazy download.
async fn materialize_metadata_tree(
    remote_source: &RemoteDatasetSource,
    cache_root: &Path,
    cache_access: ObjectStoreCacheAccess,
    progress_reporter: &DatasetOpenProgressReporter<'_>,
) -> Result<(), LerobotError> {
    let mut report_files_discovered = |objects_discovered| {
        progress_reporter(DatasetOpenProgress::MetadataObjectListingAdvanced {
            objects_discovered,
        });
    };
    let mut metadata_keys = remote_source
        .file_provider()
        .list_files("meta/", &mut report_files_discovered)
        .await
        .map_err(|list_error| LerobotError::Source {
            context: format!("listing meta/ under {}", remote_source.identifier()),
            detail: list_error.to_string(),
        })?;
    metadata_keys.sort();
    if metadata_keys.is_empty() {
        return Err(LerobotError::Io {
            path: cache_root.join("meta"),
            source: io::Error::new(
                io::ErrorKind::NotFound,
                format!("no meta/ objects found at {}", remote_source.identifier()),
            ),
        });
    }
    let total_objects = u64::try_from(metadata_keys.len()).unwrap_or(u64::MAX);
    progress_reporter(DatasetOpenProgress::MetadataObjectsListed { total_objects });
    for (metadata_object_index, metadata_key) in metadata_keys.into_iter().enumerate() {
        let object_ordinal = u64::try_from(metadata_object_index)
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let local_metadata_path = safe_dataset_relative_path(cache_root, &metadata_key)?;
        download_metadata_object_with_progress(
            remote_source,
            &metadata_key,
            &local_metadata_path,
            cache_root,
            cache_access,
            object_ordinal,
            total_objects,
            progress_reporter,
        )
        .await?;
        progress_reporter(DatasetOpenProgress::MetadataObjectDownloaded {
            completed_objects: object_ordinal,
            total_objects,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn download_metadata_object_with_progress(
    remote_source: &RemoteDatasetSource,
    relative_key: &str,
    local_path: &Path,
    cache_root: &Path,
    cache_access: ObjectStoreCacheAccess,
    object_ordinal: u64,
    total_objects: u64,
    progress_reporter: &DatasetOpenProgressReporter<'_>,
) -> Result<(), LerobotError> {
    let mut partial_path_text = local_path.as_os_str().to_owned();
    partial_path_text.push(".partial");
    let partial_path = PathBuf::from(partial_path_text);
    let download = download_object_to(
        remote_source,
        relative_key,
        local_path,
        cache_root,
        cache_access,
    );
    tokio::pin!(download);
    let mut last_observed_bytes = None;
    let mut observation_interval =
        tokio::time::interval(DATASET_OPEN_PROGRESS_OBSERVATION_INTERVAL);
    observation_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    observation_interval.tick().await;

    loop {
        tokio::select! {
            download_result = &mut download => return download_result,
            _ = observation_interval.tick() => {
                let observed_bytes = match tokio::fs::metadata(&partial_path).await {
                    Ok(metadata) => Some(metadata.len()),
                    Err(metadata_error) if metadata_error.kind() == io::ErrorKind::NotFound => {
                        tokio::fs::metadata(local_path).await.ok().map(|metadata| metadata.len())
                    }
                    Err(_) => None,
                };
                let Some(observed_bytes) = observed_bytes else {
                    continue;
                };
                if last_observed_bytes == Some(observed_bytes) {
                    continue;
                }
                last_observed_bytes = Some(observed_bytes);
                progress_reporter(DatasetOpenProgress::MetadataObjectDownloadAdvanced {
                    object_ordinal,
                    total_objects,
                    bytes_downloaded: observed_bytes,
                });
            }
        }
    }
}

/// Reads the object at `relative_key` from the source and writes it to
/// `local_path` (creating parent directories). A missing object is a hard error
/// (the metadata claimed it exists), mirroring a local `File::open` failure.
async fn download_object_to(
    remote_source: &RemoteDatasetSource,
    relative_key: &str,
    local_path: &Path,
    cache_root: &Path,
    cache_access: ObjectStoreCacheAccess,
) -> Result<(), LerobotError> {
    if cache_access == ObjectStoreCacheAccess::SharedReadOnly {
        prepare_shared_cache_destination_async(cache_root, local_path).await?;
    }

    // STREAM the object straight to disk (get_to_file creates parent dirs and
    // atomically renames on success) instead of reading the whole payload into
    // memory: v3.0 video chunk files are hundreds of MB, and buffering them x
    // download concurrency would spike memory badly on an object-store source.
    let materialization_outcome = remote_source
        .file_provider()
        .materialize_file(relative_key, local_path)
        .await
        .map_err(|read_error| LerobotError::Source {
            context: format!(
                "downloading {relative_key} from {}",
                remote_source.identifier()
            ),
            detail: read_error.to_string(),
        })?;
    match materialization_outcome {
        DatasetFileMaterialization::Materialized => {
            if cache_access == ObjectStoreCacheAccess::SharedReadOnly {
                set_shared_file_permissions_async(local_path).await?;
            }
            Ok(())
        }
        // A dataset that references an object which does not exist is
        // unrecoverable for this read, so the not-found OUTCOME becomes an error
        // here (unlike the storage layer, where it is success-side).
        DatasetFileMaterialization::NotFound => Err(LerobotError::Io {
            path: local_path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "object {relative_key} not found at {}",
                    remote_source.identifier()
                ),
            ),
        }),
    }
}

async fn prepare_shared_cache_destination_async(
    cache_root: &Path,
    local_path: &Path,
) -> Result<(), LerobotError> {
    let owned_cache_root = cache_root.to_path_buf();
    let owned_local_path = local_path.to_path_buf();
    let task_local_path = owned_local_path.clone();
    tokio::task::spawn_blocking(move || {
        prepare_shared_cache_destination(&owned_cache_root, &owned_local_path)
    })
    .await
    .map_err(|join_error| LerobotError::Io {
        path: task_local_path,
        source: io::Error::other(format!(
            "shared cache destination preparation task failed: {join_error}"
        )),
    })?
}

async fn set_shared_file_permissions_async(local_path: &Path) -> Result<(), LerobotError> {
    let owned_local_path = local_path.to_path_buf();
    let task_local_path = owned_local_path.clone();
    tokio::task::spawn_blocking(move || {
        set_shared_file_permissions(&owned_local_path).map_err(|source| LerobotError::Io {
            path: owned_local_path,
            source,
        })
    })
    .await
    .map_err(|join_error| LerobotError::Io {
        path: task_local_path,
        source: io::Error::other(format!(
            "shared cache file permission task failed: {join_error}"
        )),
    })?
}

fn prepare_shared_cache_destination(
    cache_root: &Path,
    local_path: &Path,
) -> Result<(), LerobotError> {
    let parent_directory = local_path.parent().ok_or_else(|| LerobotError::Io {
        path: local_path.to_path_buf(),
        source: io::Error::new(
            io::ErrorKind::InvalidInput,
            "cached object path has no parent directory",
        ),
    })?;
    parent_directory
        .strip_prefix(cache_root)
        .map_err(|_| LerobotError::Io {
            path: local_path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::InvalidInput,
                "cached object path is outside its cache root",
            ),
        })?;

    fs::create_dir_all(parent_directory).map_err(|source| LerobotError::Io {
        path: parent_directory.to_path_buf(),
        source,
    })?;

    let mut directory_to_share = Some(parent_directory);
    while let Some(directory) = directory_to_share {
        set_shared_directory_permissions(directory).map_err(|source| LerobotError::Io {
            path: directory.to_path_buf(),
            source,
        })?;
        if directory == cache_root {
            break;
        }
        directory_to_share = directory.parent();
    }
    Ok(())
}

#[cfg(unix)]
fn set_shared_directory_permissions(directory: &Path) -> io::Result<()> {
    let mut permissions = fs::metadata(directory)?.permissions();
    permissions.set_mode(permissions.mode() | 0o055);
    fs::set_permissions(directory, permissions)
}

#[cfg(not(unix))]
fn set_shared_directory_permissions(_directory: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_shared_file_permissions(file: &Path) -> io::Result<()> {
    let mut permissions = fs::metadata(file)?.permissions();
    permissions.set_mode(permissions.mode() | 0o044);
    fs::set_permissions(file, permissions)
}

#[cfg(not(unix))]
fn set_shared_file_permissions(_file: &Path) -> io::Result<()> {
    Ok(())
}

/// Renders a path relative to the dataset root as a forward-slash object key
/// (object stores are always `/`-delimited, even when the reader ran on
/// Windows).
fn to_forward_slash_key(relative: &Path) -> String {
    relative
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>()
        .join("/")
}

/// Opens a parquet file for arrow reading with path-annotated errors.
///
/// # Errors
///
/// Returns [`LerobotError::Io`] when `path` cannot be opened or
/// [`LerobotError::Parquet`] when its metadata cannot be decoded.
pub fn open_parquet_reader_builder(
    path: &Path,
) -> Result<ParquetRecordBatchReaderBuilder<File>, LerobotError> {
    let file = File::open(path).map_err(|source| LerobotError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    ParquetRecordBatchReaderBuilder::try_new(file).map_err(|source| LerobotError::Parquet {
        path: path.to_path_buf(),
        source,
    })
}

/// Enumerates `<base_dir>/chunk-*/file-*.parquet` in (chunk, file) numeric
/// order. Non-matching entries are skipped with a warning.
///
/// # Errors
///
/// Returns [`LerobotError::Io`] when a required directory cannot be read.
pub fn enumerate_chunked_parquet_files(base_dir: &Path) -> Result<Vec<PathBuf>, LerobotError> {
    let mut chunk_dirs: Vec<(u64, PathBuf)> = Vec::new();
    for entry_result in fs::read_dir(base_dir).map_err(|source| LerobotError::Io {
        path: base_dir.to_path_buf(),
        source,
    })? {
        let entry = entry_result.map_err(|source| LerobotError::Io {
            path: base_dir.to_path_buf(),
            source,
        })?;
        let entry_name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_dir()
            && let Some(chunk_number) = parse_indexed_name(&entry_name, "chunk-", "")
        {
            chunk_dirs.push((chunk_number, entry.path()));
            continue;
        }
        tracing::warn!(entry = entry_name, dir = %base_dir.display(), "skipping unrecognized entry");
    }
    chunk_dirs.sort_by_key(|(chunk_number, _)| *chunk_number);

    let mut parquet_files = Vec::new();
    for (_, chunk_dir) in chunk_dirs {
        let mut files_in_chunk: Vec<(u64, PathBuf)> = Vec::new();
        for entry_result in fs::read_dir(&chunk_dir).map_err(|source| LerobotError::Io {
            path: chunk_dir.clone(),
            source,
        })? {
            let entry = entry_result.map_err(|source| LerobotError::Io {
                path: chunk_dir.clone(),
                source,
            })?;
            let entry_name = entry.file_name().to_string_lossy().into_owned();
            if let Some(file_number) = parse_indexed_name(&entry_name, "file-", ".parquet") {
                files_in_chunk.push((file_number, entry.path()));
            } else {
                tracing::warn!(entry = entry_name, dir = %chunk_dir.display(), "skipping unrecognized entry");
            }
        }
        files_in_chunk.sort_by_key(|(file_number, _)| *file_number);
        parquet_files.extend(files_in_chunk.into_iter().map(|(_, path)| path));
    }
    Ok(parquet_files)
}

fn parse_indexed_name(name: &str, prefix: &str, suffix: &str) -> Option<u64> {
    name.strip_prefix(prefix)?
        .strip_suffix(suffix)?
        .parse()
        .ok()
}

fn read_all_episode_metadata(
    root: &Path,
    info: &DatasetInfo,
    video_keys: &[String],
    progress_reporter: &DatasetOpenProgressReporter<'_>,
) -> Result<Vec<Episode>, LerobotError> {
    let episodes_dir = root.join("meta").join("episodes");
    let parquet_files = enumerate_chunked_parquet_files(&episodes_dir)?;
    if parquet_files.is_empty() {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!(
                "no episode metadata parquet files found under {}",
                episodes_dir.display()
            ),
        });
    }

    let mut episodes = Vec::new();
    let mut rows_parsed = 0_u64;
    for parquet_path in &parquet_files {
        read_episodes_from_file(
            parquet_path,
            root,
            info,
            video_keys,
            &mut episodes,
            &mut rows_parsed,
            progress_reporter,
        )?;
    }

    episodes.sort_by_key(|episode| episode.episode_index);
    for adjacent in episodes.windows(2) {
        if adjacent[0].episode_index == adjacent[1].episode_index {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "duplicate episode_index {} in episode metadata",
                    adjacent[0].episode_index
                ),
            });
        }
    }
    Ok(episodes)
}

fn read_episodes_from_file(
    parquet_path: &Path,
    root: &Path,
    info: &DatasetInfo,
    video_keys: &[String],
    episodes: &mut Vec<Episode>,
    rows_parsed: &mut u64,
    progress_reporter: &DatasetOpenProgressReporter<'_>,
) -> Result<(), LerobotError> {
    let builder = open_parquet_reader_builder(parquet_path)?;

    // The stats/* columns are deeply nested lists that the reader never needs;
    // project them away so schema handling stays simple and reads stay cheap.
    let non_stats_root_indices: Vec<usize> = builder
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| !field.name().starts_with("stats/"))
        .map(|(position, _)| position)
        .collect();
    let projection = ProjectionMask::roots(builder.parquet_schema(), non_stats_root_indices);

    let reader = builder
        .with_projection(projection)
        .build()
        .map_err(|source| LerobotError::Parquet {
            path: parquet_path.to_path_buf(),
            source,
        })?;

    for batch_result in reader {
        let batch = batch_result?;
        *rows_parsed =
            rows_parsed.saturating_add(u64::try_from(batch.num_rows()).unwrap_or(u64::MAX));
        for row in 0..batch.num_rows() {
            episodes.push(episode_from_row(&batch, row, root, info, video_keys)?);
        }
        progress_reporter(DatasetOpenProgress::EpisodeMetadataRowsParsed {
            rows_parsed: *rows_parsed,
        });
    }
    Ok(())
}

/// The columns that together locate one episode's window into a camera's
/// packed video file. `LeRobot` writes the whole group for every camera on
/// every episode, so the group is also the unit of ABSENCE.
const VIDEO_WINDOW_COLUMN_SUFFIXES: [&str; 4] = [
    "chunk_index",
    "file_index",
    "from_timestamp",
    "to_timestamp",
];

/// Reads one episode's window into `video_key`'s packed file, or `None` when
/// the episode declares no footage for that camera.
///
/// A fully-null window group means absence. That is the only vocabulary v3 has
/// for it: the schema is fixed per dataset, so an episode cannot simply omit a
/// camera's columns. Refusing to read the null (the previous behavior) made a
/// dataset with one partially-recorded camera fail to OPEN at all, which is a
/// harsher verdict than the data warrants.
///
/// A PARTIALLY null group is still an error. It is indistinguishable from a
/// truncated or half-written row, and inferring absence from it would let a
/// writer bug disappear as a silently dropped camera.
fn video_segment_from_row(
    batch: &RecordBatch,
    row: usize,
    root: &Path,
    info: &DatasetInfo,
    video_key: &str,
    episode_index: u32,
) -> Result<Option<VideoSegment>, LerobotError> {
    let mut null_column_count = 0_usize;
    for column_suffix in VIDEO_WINDOW_COLUMN_SUFFIXES {
        if row_is_null(batch, row, &format!("videos/{video_key}/{column_suffix}"))? {
            null_column_count += 1;
        }
    }
    if null_column_count == VIDEO_WINDOW_COLUMN_SUFFIXES.len() {
        return Ok(None);
    }
    if null_column_count != 0 {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!(
                "episode {episode_index} nulls {null_column_count} of the \
                 {total_columns} \"videos/{video_key}/*\" window columns; a camera is \
                 absent only when the whole group is null",
                total_columns = VIDEO_WINDOW_COLUMN_SUFFIXES.len()
            ),
        });
    }

    let chunk_index = u32_row(batch, row, &format!("videos/{video_key}/chunk_index"))?;
    let file_index = u32_row(batch, row, &format!("videos/{video_key}/file_index"))?;
    let relative_video_path = render_path_template(
        info.video_path_template(),
        Some(video_key),
        chunk_index,
        file_index,
    )?;
    let timestamp_range = TimestampRange::new(
        f64_row(batch, row, &format!("videos/{video_key}/from_timestamp"))?,
        f64_row(batch, row, &format!("videos/{video_key}/to_timestamp"))?,
    )
    .map_err(|source| LerobotError::InconsistentMetadata {
        detail: format!(
            "episode {episode_index} camera {video_key:?} has an invalid video range: {source}"
        ),
    })?;
    Ok(Some(VideoSegment {
        file: relative_video_path.join_under(root),
        chunk_index,
        file_index,
        timestamp_range,
    }))
}

fn episode_from_row(
    batch: &RecordBatch,
    row: usize,
    root: &Path,
    info: &DatasetInfo,
    video_keys: &[String],
) -> Result<Episode, LerobotError> {
    let episode_index = u32_row(batch, row, "episode_index")?;
    let mut video_segments = BTreeMap::new();
    for video_key in video_keys {
        if let Some(segment) =
            video_segment_from_row(batch, row, root, info, video_key, episode_index)?
        {
            video_segments.insert(video_key.clone(), segment);
        }
    }

    Ok(Episode {
        episode_index,
        length_frames: u64_row(batch, row, "length")?,
        tasks: tasks_from_row(batch, row)?,
        video_segments,
        dataset_from_index: u64_row(batch, row, "dataset_from_index")?,
        dataset_to_index: u64_row(batch, row, "dataset_to_index")?,
        data_chunk_index: u32_row(batch, row, "data/chunk_index")?,
        data_file_index: u32_row(batch, row, "data/file_index")?,
    })
}

/// Resolves a dataset-relative path while rejecting absolute paths and
/// traversal components.
///
/// # Errors
///
/// Returns [`LerobotError::InconsistentMetadata`] when `relative_path` is
/// empty, absolute, or contains a non-normal component.
pub fn safe_dataset_relative_path(
    root: &Path,
    relative_path: &str,
) -> Result<PathBuf, LerobotError> {
    DatasetRelativePath::parse(relative_path.to_owned())
        .map(|relative_path| relative_path.join_under(root))
        .map_err(|source| LerobotError::InconsistentMetadata {
            detail: source.to_string(),
        })
}

fn tasks_from_row(batch: &RecordBatch, row: usize) -> Result<Vec<String>, LerobotError> {
    let tasks_column = require_list_column(batch, "tasks")?;
    if tasks_column.is_null(row) {
        return Ok(Vec::new());
    }
    let row_values = tasks_column.value(row);
    let task_strings = row_values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| LerobotError::InconsistentMetadata {
            detail: "column \"tasks\" is not a list of utf8 strings".to_owned(),
        })?;
    Ok((0..task_strings.len())
        .filter(|&position| !task_strings.is_null(position))
        .map(|position| task_strings.value(position).to_owned())
        .collect())
}

fn validate_dataset_invariants(
    info: &DatasetInfo,
    episodes: &[Episode],
) -> Result<(), LerobotError> {
    let episode_count =
        u32::try_from(episodes.len()).map_err(|_| LerobotError::InconsistentMetadata {
            detail: "episode count exceeds u32 range".to_owned(),
        })?;
    if episode_count != info.total_episodes() {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!(
                "info.json declares {} episodes but episode metadata has {episode_count}",
                info.total_episodes()
            ),
        });
    }

    let summed_frames: u64 = episodes.iter().map(|episode| episode.length_frames).sum();
    if summed_frames != info.total_frames() {
        return Err(LerobotError::InconsistentMetadata {
            detail: format!(
                "info.json declares {} total frames but episode lengths sum to {summed_frames}",
                info.total_frames()
            ),
        });
    }

    for episode in episodes {
        let row_range_length = episode
            .dataset_to_index
            .checked_sub(episode.dataset_from_index);
        if row_range_length != Some(episode.length_frames) {
            return Err(LerobotError::InconsistentMetadata {
                detail: format!(
                    "episode {} has length {} but dataset row range {}..{}",
                    episode.episode_index,
                    episode.length_frames,
                    episode.dataset_from_index,
                    episode.dataset_to_index
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Dataset` crosses threads (async opens parse metadata on a blocking
    /// thread; export moves an owned handle into `spawn_blocking`), so it must
    /// stay `Send + Sync`. This fails to compile if a field regresses that.
    #[test]
    fn dataset_stays_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dataset>();
    }

    #[test]
    fn object_store_cache_cleanup_waits_for_the_last_owner() {
        let temp_directory = tempfile::tempdir().expect("cache tempdir");
        let cache_path = temp_directory.path().to_path_buf();
        fs::write(cache_path.join("cached-object"), b"payload").expect("write cached object");
        let first_cache_owner = Arc::new(ObjectStoreCache::from_temp_directory(temp_directory));
        let last_cache_owner = Arc::clone(&first_cache_owner);

        drop(first_cache_owner);
        assert!(
            cache_path.exists(),
            "the cache must remain while another dataset owner exists"
        );
        drop(last_cache_owner);

        let cleanup_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while cache_path.exists() && std::time::Instant::now() < cleanup_deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            !cache_path.exists(),
            "the dedicated cleanup worker must eventually remove an unowned cache"
        );
    }

    #[test]
    fn open_missing_directory_is_io_error() {
        let error = Dataset::open("/nonexistent/definitely-not-a-dataset").unwrap_err();
        assert!(matches!(error, LerobotError::Io { .. }));
    }

    #[test]
    fn open_directory_without_meta_is_io_error() {
        let empty_dir = tempfile::tempdir().unwrap();
        let error = Dataset::open(empty_dir.path()).unwrap_err();
        assert!(matches!(error, LerobotError::Io { .. }));
    }

    #[test]
    fn open_v2_dataset_is_unsupported_version_error() {
        let dataset_dir = tempfile::tempdir().unwrap();
        let meta_dir = dataset_dir.path().join("meta");
        std::fs::create_dir_all(&meta_dir).unwrap();
        std::fs::write(
            meta_dir.join("info.json"),
            r#"{
                "codebase_version": "v2.1",
                "robot_type": "so100",
                "total_episodes": 1,
                "total_frames": 10,
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {}
            }"#,
        )
        .unwrap();
        let error = Dataset::open(dataset_dir.path()).unwrap_err();
        assert!(
            matches!(error, LerobotError::UnsupportedVersion { ref found } if found == "v2.1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn open_v3_dataset_without_episode_metadata_is_io_error() {
        let dataset_dir = tempfile::tempdir().unwrap();
        let meta_dir = dataset_dir.path().join("meta");
        std::fs::create_dir_all(&meta_dir).unwrap();
        std::fs::write(
            meta_dir.join("info.json"),
            r#"{
                "codebase_version": "v3.0",
                "robot_type": "so100",
                "total_episodes": 1,
                "total_frames": 10,
                "fps": 30,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {}
            }"#,
        )
        .unwrap();
        // meta/episodes does not exist.
        let error = Dataset::open(dataset_dir.path()).unwrap_err();
        assert!(matches!(error, LerobotError::Io { .. }));
    }

    #[test]
    fn parse_indexed_name_accepts_only_exact_pattern() {
        assert_eq!(parse_indexed_name("chunk-000", "chunk-", ""), Some(0));
        assert_eq!(parse_indexed_name("chunk-1234", "chunk-", ""), Some(1234));
        assert_eq!(
            parse_indexed_name("file-007.parquet", "file-", ".parquet"),
            Some(7)
        );
        assert_eq!(
            parse_indexed_name("file-007.mp4", "file-", ".parquet"),
            None
        );
        assert_eq!(parse_indexed_name("chunk-invalid", "chunk-", ""), None);
        assert_eq!(parse_indexed_name("other", "chunk-", ""), None);
    }

    #[cfg(unix)]
    #[test]
    fn shared_object_store_cache_is_traversable_and_readable_by_another_user() {
        let cache_parent_owner = tempfile::tempdir().expect("cache parent owner");
        let shared_cache_parent = cache_parent_owner.path().join("shared-cache");
        fs::create_dir(&shared_cache_parent).expect("create shared cache parent");
        fs::set_permissions(&shared_cache_parent, fs::Permissions::from_mode(0o700))
            .expect("make initial parent private");

        let (cache, access) = create_shared_object_store_cache(
            &shared_cache_parent,
            "test-dataset-",
            "test shared cache directory",
        )
        .expect("create shared object-store cache");
        assert_eq!(access, ObjectStoreCacheAccess::SharedReadOnly);
        assert_eq!(cache.path().parent(), Some(shared_cache_parent.as_path()));
        assert_eq!(permission_mode(&shared_cache_parent), 0o755);
        assert_eq!(permission_mode(cache.path()), 0o755);

        let cached_video_path = cache
            .path()
            .join("videos")
            .join("observation.images.front")
            .join("chunk-000")
            .join("file-000.mp4");
        prepare_shared_cache_destination(cache.path(), &cached_video_path)
            .expect("prepare shared video destination");
        fs::write(&cached_video_path, b"video bytes").expect("write cached video");
        fs::set_permissions(&cached_video_path, fs::Permissions::from_mode(0o600))
            .expect("make initial video private");
        set_shared_file_permissions(&cached_video_path).expect("share cached video");

        assert_eq!(permission_mode(&cached_video_path), 0o644);
        let mut ancestor = cached_video_path.parent();
        while let Some(directory) = ancestor {
            assert_eq!(permission_mode(directory), 0o755);
            if directory == cache.path() {
                break;
            }
            ancestor = directory.parent();
        }
    }

    #[cfg(unix)]
    fn permission_mode(path: &Path) -> u32 {
        fs::metadata(path)
            .expect("path metadata")
            .permissions()
            .mode()
            & 0o777
    }
}
