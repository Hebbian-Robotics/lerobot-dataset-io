//! Provider-neutral description of where a dataset's raw files live.
//!
//! A `LeRobot` v3 dataset is a tree of `meta/`, `data/`, and `videos/` files.
//! [`DatasetSource`] represents either a local path or a remote location whose
//! files are exposed through [`DatasetFileProvider`]. URI parsing, credentials,
//! and concrete storage clients belong to adapters; the dataset reader only
//! branches on this typed source. The local variant keeps the exact filesystem
//! fast path; the remote variant materializes metadata (and, on demand,
//! the video and data files ffmpeg/parquet need) into a local temp cache.

use std::path::PathBuf;
use std::sync::Arc;

use crate::DatasetFileProvider;

/// Local materialization policy for a remotely backed dataset.
#[derive(Debug, Clone, Default)]
pub enum DatasetCachePolicy {
    /// Give each opened dataset its own automatically removed temporary cache.
    #[default]
    PrivateTemporaryDirectory,
    /// Materialize beneath a caller-selected shared directory with read-only
    /// permissions suitable for reuse across processes or users.
    SharedReadOnlyDirectory {
        /// Parent directory under which the temporary cache is created.
        parent_directory: PathBuf,
        /// Prefix used for the generated cache-directory name.
        temporary_directory_prefix: String,
        /// Human-readable setting name included in configuration errors.
        configuration_name: String,
    },
}

/// The parsed, validated location of a dataset's raw files.
///
/// Constructed or parsed at the system boundary. `Local` uses the on-disk fast
/// path; `Remote` materializes files through a provider-neutral interface.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    /// A directory on the local filesystem (also the target of a `file://`
    /// URI). Read with plain `std::fs` — no runtime or network required.
    Local(PathBuf),
    /// A dataset whose files are materialized from a remote provider.
    Remote(RemoteDatasetSource),
}

/// A remotely backed dataset location: its file provider plus a stable source
/// identity.
///
/// Cheap to clone because the provider is shared through [`Arc`]. All remote
/// reads are async and awaited on the caller's runtime; there is no synchronous
/// download path.
#[derive(Debug, Clone)]
pub struct RemoteDatasetSource {
    file_provider: Arc<dyn DatasetFileProvider>,
    /// Stable, log-safe identity such as `gs://bucket/prefix`; what the
    /// registry and index manifest record so the dataset can be reopened.
    identifier: String,
    cache_policy: DatasetCachePolicy,
}

impl RemoteDatasetSource {
    /// Provider used to enumerate and materialize this dataset's files.
    #[must_use]
    pub fn file_provider(&self) -> &dyn DatasetFileProvider {
        self.file_provider.as_ref()
    }

    /// Local materialization policy selected for this remote source.
    #[must_use]
    pub fn cache_policy(&self) -> &DatasetCachePolicy {
        &self.cache_policy
    }

    /// Stable, log-safe identity (for example `gs://bucket/prefix`).
    #[must_use]
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
}

impl DatasetSource {
    /// Wraps a remote dataset-file provider with its stable, log-safe source
    /// identity.
    #[must_use]
    pub fn from_file_provider(
        file_provider: Arc<dyn DatasetFileProvider>,
        source_identifier: impl Into<String>,
    ) -> Self {
        Self::from_file_provider_with_cache_policy(
            file_provider,
            source_identifier,
            DatasetCachePolicy::default(),
        )
    }

    /// Wraps a remote file provider using an explicit local cache policy.
    #[must_use]
    pub fn from_file_provider_with_cache_policy(
        file_provider: Arc<dyn DatasetFileProvider>,
        source_identifier: impl Into<String>,
        cache_policy: DatasetCachePolicy,
    ) -> Self {
        Self::Remote(RemoteDatasetSource {
            file_provider,
            identifier: source_identifier.into(),
            cache_policy,
        })
    }

    /// A stable, log-safe identifier for this source: the local path or the
    /// remote identifier. This is what the registry and index manifest record so
    /// the dataset can be reopened later.
    #[must_use]
    pub fn identifier(&self) -> String {
        match self {
            Self::Local(path) => path.to_string_lossy().into_owned(),
            Self::Remote(remote_source) => remote_source.identifier.clone(),
        }
    }

    /// Human/log-friendly description (no secrets); same as [`Self::identifier`].
    #[must_use]
    pub fn describe(&self) -> String {
        self.identifier()
    }
}
