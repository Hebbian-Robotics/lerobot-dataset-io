//! Adapter from Apache Arrow's `object_store` ecosystem to the narrow dataset
//! file-provider contract.

use std::path::Path as FileSystemPath;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt as _};
use tokio::io::AsyncWriteExt as _;

use crate::{
    DatasetCachePolicy, DatasetFileMaterialization, DatasetFileProvider, DatasetFileProviderError,
    DatasetSource,
};

/// Failure to construct an [`object_store`]-backed [`DatasetSource`] from a
/// URL and backend options.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ObjectStoreDatasetSourceError {
    /// The supplied source identifier is not a valid URL.
    #[error("invalid object-store dataset URL: {source}")]
    InvalidUrl {
        /// URL parser failure.
        #[source]
        source: url::ParseError,
    },
    /// The URL scheme or backend options could not construct an object store.
    #[error("cannot open object-store dataset URL: {source}")]
    ObjectStore {
        /// Backend-construction failure.
        #[source]
        source: object_store::Error,
    },
}

/// A dataset-file provider backed by any Apache Arrow [`ObjectStore`].
///
/// `dataset_prefix` is the object path corresponding to the dataset root.
/// Paths exposed through [`DatasetFileProvider`] are always relative to that
/// root.
#[derive(Debug)]
pub struct ObjectStoreDatasetFileProvider {
    object_store: Arc<dyn ObjectStore>,
    dataset_prefix: ObjectPath,
}

impl ObjectStoreDatasetFileProvider {
    /// Creates a provider over `object_store` rooted at `dataset_prefix`.
    #[must_use]
    pub fn new(object_store: Arc<dyn ObjectStore>, dataset_prefix: ObjectPath) -> Self {
        Self {
            object_store,
            dataset_prefix,
        }
    }

    /// Object path corresponding to the dataset root.
    #[must_use]
    pub fn dataset_prefix(&self) -> &ObjectPath {
        &self.dataset_prefix
    }

    fn object_location(
        &self,
        dataset_relative_path: &str,
    ) -> Result<ObjectPath, DatasetFileProviderError> {
        let relative_path = ObjectPath::parse(dataset_relative_path)
            .map_err(|source| Box::new(source) as DatasetFileProviderError)?;
        Ok(self
            .dataset_prefix
            .parts()
            .chain(relative_path.parts())
            .collect())
    }
}

#[async_trait]
impl DatasetFileProvider for ObjectStoreDatasetFileProvider {
    async fn list_files(
        &self,
        relative_prefix: &str,
        report_files_discovered: &mut (dyn FnMut(u64) + Send),
    ) -> Result<Vec<String>, DatasetFileProviderError> {
        let object_prefix = self.object_location(relative_prefix)?;
        let mut listed_objects = self.object_store.list(Some(&object_prefix));
        let mut relative_paths = Vec::new();
        let mut files_discovered = 0_u64;

        while let Some(object_result) = listed_objects.next().await {
            let object_metadata =
                object_result.map_err(|source| Box::new(source) as DatasetFileProviderError)?;
            let relative_parts = object_metadata
                .location
                .prefix_match(&self.dataset_prefix)
                .ok_or_else(|| {
                    Box::new(std::io::Error::other(format!(
                        "object-store listing returned path {:?} outside dataset prefix {:?}",
                        object_metadata.location, self.dataset_prefix
                    ))) as DatasetFileProviderError
                })?;
            let relative_path: ObjectPath = relative_parts.collect();
            if relative_path.is_root() {
                continue;
            }
            relative_paths.push(relative_path.to_string());
            files_discovered = files_discovered.saturating_add(1);
            report_files_discovered(files_discovered);
        }

        Ok(relative_paths)
    }

    async fn materialize_file(
        &self,
        relative_path: &str,
        destination_path: &FileSystemPath,
    ) -> Result<DatasetFileMaterialization, DatasetFileProviderError> {
        let object_location = self.object_location(relative_path)?;
        let get_result = match self.object_store.get(&object_location).await {
            Ok(get_result) => get_result,
            Err(object_store::Error::NotFound { .. }) => {
                return Ok(DatasetFileMaterialization::NotFound);
            }
            Err(source) => return Err(Box::new(source)),
        };

        let destination_parent = destination_path.parent().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "materialization destination {} has no parent directory",
                    destination_path.display()
                ),
            )) as DatasetFileProviderError
        })?;
        tokio::fs::create_dir_all(destination_parent)
            .await
            .map_err(|source| Box::new(source) as DatasetFileProviderError)?;

        let temporary_file = tempfile::Builder::new()
            .prefix(".lerobot-materialization-")
            .tempfile_in(destination_parent)
            .map_err(|source| Box::new(source) as DatasetFileProviderError)?;
        let asynchronous_file = temporary_file
            .reopen()
            .map(tokio::fs::File::from_std)
            .map_err(|source| Box::new(source) as DatasetFileProviderError)?;
        let mut asynchronous_file = asynchronous_file;
        let mut payload_stream = get_result.into_stream();
        while let Some(payload_result) = payload_stream.next().await {
            let payload =
                payload_result.map_err(|source| Box::new(source) as DatasetFileProviderError)?;
            asynchronous_file
                .write_all(&payload)
                .await
                .map_err(|source| Box::new(source) as DatasetFileProviderError)?;
        }
        asynchronous_file
            .sync_all()
            .await
            .map_err(|source| Box::new(source) as DatasetFileProviderError)?;
        drop(asynchronous_file);

        temporary_file
            .persist(destination_path)
            .map_err(|source| Box::new(source) as DatasetFileProviderError)?;
        Ok(DatasetFileMaterialization::Materialized)
    }
}

impl DatasetSource {
    /// Constructs a remote dataset source from any Apache Arrow object store.
    #[must_use]
    pub fn from_object_store(
        object_store: Arc<dyn ObjectStore>,
        dataset_prefix: ObjectPath,
        source_identifier: impl Into<String>,
    ) -> Self {
        Self::from_file_provider(
            Arc::new(ObjectStoreDatasetFileProvider::new(
                object_store,
                dataset_prefix,
            )),
            source_identifier,
        )
    }

    /// Constructs an object-store source using an explicit local cache policy.
    #[must_use]
    pub fn from_object_store_with_cache_policy(
        object_store: Arc<dyn ObjectStore>,
        dataset_prefix: ObjectPath,
        source_identifier: impl Into<String>,
        cache_policy: DatasetCachePolicy,
    ) -> Self {
        Self::from_file_provider_with_cache_policy(
            Arc::new(ObjectStoreDatasetFileProvider::new(
                object_store,
                dataset_prefix,
            )),
            source_identifier,
            cache_policy,
        )
    }

    /// Constructs a source from a URL supported by the enabled `object_store`
    /// backend features and its provider-specific options.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectStoreDatasetSourceError`] when the URL is malformed,
    /// its scheme is not enabled, or its options cannot configure the backend.
    pub fn from_object_store_url<Options, Key, Value>(
        source_url: &str,
        options: Options,
    ) -> Result<Self, ObjectStoreDatasetSourceError>
    where
        Options: IntoIterator<Item = (Key, Value)>,
        Key: AsRef<str>,
        Value: Into<String>,
    {
        Self::from_object_store_url_with_cache_policy(
            source_url,
            options,
            DatasetCachePolicy::default(),
        )
    }

    /// URL constructor with an explicit local materialization policy.
    ///
    /// # Errors
    ///
    /// Returns [`ObjectStoreDatasetSourceError`] when the URL is malformed,
    /// its scheme is not enabled, or its options cannot configure the backend.
    pub fn from_object_store_url_with_cache_policy<Options, Key, Value>(
        source_url: &str,
        options: Options,
        cache_policy: DatasetCachePolicy,
    ) -> Result<Self, ObjectStoreDatasetSourceError>
    where
        Options: IntoIterator<Item = (Key, Value)>,
        Key: AsRef<str>,
        Value: Into<String>,
    {
        let parsed_url = url::Url::parse(source_url)
            .map_err(|source| ObjectStoreDatasetSourceError::InvalidUrl { source })?;
        let source_identifier = sanitized_source_identifier(&parsed_url);
        let (object_store, dataset_prefix) = object_store::parse_url_opts(&parsed_url, options)
            .map_err(|source| ObjectStoreDatasetSourceError::ObjectStore { source })?;
        Ok(Self::from_object_store_with_cache_policy(
            Arc::from(object_store),
            dataset_prefix,
            source_identifier,
            cache_policy,
        ))
    }
}

/// Removes URL components that commonly carry credentials or request-scoped
/// secrets before the identity reaches logs or persisted manifests.
fn sanitized_source_identifier(parsed_url: &url::Url) -> String {
    let mut sanitized_url = parsed_url.clone();
    let _ = sanitized_url.set_username("");
    let _ = sanitized_url.set_password(None);
    sanitized_url.set_query(None);
    sanitized_url.set_fragment(None);
    sanitized_url.to_string()
}

#[cfg(test)]
mod tests {
    use object_store::ObjectStoreExt as _;
    use object_store::PutPayload;
    use object_store::memory::InMemory;

    use super::*;

    #[test]
    fn source_identifiers_and_constructor_errors_do_not_expose_url_secrets() {
        let parsed_url = url::Url::parse(
            "s3://access-key:secret-password@example-bucket/dataset?token=query-secret#fragment-secret",
        )
        .expect("URL with credentials");
        let source_identifier = sanitized_source_identifier(&parsed_url);
        assert_eq!(source_identifier, "s3://example-bucket/dataset");

        let invalid_url_error = DatasetSource::from_object_store_url(
            "not-a-url?token=invalid-url-secret",
            std::iter::empty::<(&str, &str)>(),
        )
        .expect_err("relative URL must be rejected")
        .to_string();
        assert!(!invalid_url_error.contains("invalid-url-secret"));
    }

    #[tokio::test]
    async fn lists_relative_paths_and_materializes_objects_atomically() {
        let object_store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        object_store
            .put(
                &ObjectPath::from("datasets/example/meta/info.json"),
                PutPayload::from_static(b"dataset metadata"),
            )
            .await
            .expect("write test object");
        object_store
            .put(
                &ObjectPath::from("datasets/other/meta/info.json"),
                PutPayload::from_static(b"other dataset"),
            )
            .await
            .expect("write object outside dataset prefix");

        let provider =
            ObjectStoreDatasetFileProvider::new(object_store, ObjectPath::from("datasets/example"));
        let mut progress_counts = Vec::new();
        let listed_paths = provider
            .list_files("meta", &mut |files_discovered| {
                progress_counts.push(files_discovered);
            })
            .await
            .expect("list dataset files");
        assert_eq!(listed_paths, vec!["meta/info.json"]);
        assert_eq!(progress_counts, vec![1]);

        let destination_directory = tempfile::tempdir().expect("temporary destination");
        let destination_path = destination_directory.path().join("meta/info.json");
        let materialization = provider
            .materialize_file("meta/info.json", &destination_path)
            .await
            .expect("materialize object");
        assert_eq!(materialization, DatasetFileMaterialization::Materialized);
        assert_eq!(
            std::fs::read(destination_path).expect("read materialized file"),
            b"dataset metadata"
        );
    }

    #[tokio::test]
    async fn missing_object_reports_not_found_without_a_partial_destination() {
        let provider = ObjectStoreDatasetFileProvider::new(
            Arc::new(InMemory::new()),
            ObjectPath::from("datasets/example"),
        );
        let destination_directory = tempfile::tempdir().expect("temporary destination");
        let destination_path = destination_directory.path().join("missing.parquet");

        let materialization = provider
            .materialize_file("data/missing.parquet", &destination_path)
            .await
            .expect("missing object is an outcome");

        assert_eq!(materialization, DatasetFileMaterialization::NotFound);
        assert!(!destination_path.exists());
    }
}
