# lerobot-dataset-io

[![crates.io](https://img.shields.io/crates/v/lerobot-dataset-io.svg)](https://crates.io/crates/lerobot-dataset-io)
[![docs.rs](https://docs.rs/lerobot-dataset-io/badge.svg)](https://docs.rs/lerobot-dataset-io)
[![CI](https://github.com/Hebbian-Robotics/lerobot-dataset-io/actions/workflows/ci.yml/badge.svg)](https://github.com/Hebbian-Robotics/lerobot-dataset-io/actions/workflows/ci.yml)

Rust I/O for inspecting, accessing, and creating subsets of packed `LeRobot`
datasets.

The crate owns the `LeRobot` v3 format boundary. It parses `meta/info.json`,
reads packed episode metadata and per-frame Parquet columns, resolves episode
video segments, materializes remote files through a provider-neutral trait,
and writes reindexed subset datasets.

For an example of a public robotics-data application where this crate can be
used, see [Pareto](https://github.com/Hebbian-Robotics/pareto).

## Installation

```bash
cargo add lerobot-dataset-io
```

API documentation is available on
[docs.rs](https://docs.rs/lerobot-dataset-io).

## What it provides

- Local synchronous opens and async provider-backed opens.
- Ordered feature metadata with unknown fields preserved for round-trip export.
- RGB, depth-video, and unknown-video classification.
- Episode metadata and packed video-segment resolution.
- Projected and streaming reads of per-frame vector columns.
- Re-encoded or byte-preserving packed-video subset exports.
- Extension hooks for namespaced metadata and sidecars.
- Storage injection through `DatasetFileProvider`; the crate does not depend on
  a cloud vendor SDK.

The reader accepts `LeRobot` v3.x datasets. It rejects older major versions and
proceeds optimistically for newer v3 minor versions.

## Local dataset example

```rust,no_run
use lerobot_dataset_io::Dataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = Dataset::open("/datasets/example")?;

    println!(
        "{} episodes at {} fps",
        dataset.episodes().len(),
        dataset.frames_per_second().get()
    );
    for episode in dataset.episodes() {
        println!(
            "episode {}: {} frames, {} video streams",
            episode.episode_index(),
            episode.length_frames(),
            episode.video_segments().len()
        );
    }

    Ok(())
}
```

## Remote storage

Implement `DatasetFileProvider` to list dataset-relative files and atomically
materialize one file at a requested local path. Construct a `DatasetSource`
with `DatasetSource::from_file_provider`, then call `Dataset::open_source`.
This supports object stores, HTTP services, content-addressed storage, and
application-specific backends without adding them to the format crate.

The default `object-store` feature provides an adapter for Apache Arrow's
`object_store` crate, including local filesystem and in-memory stores. Enable
`aws`, `gcp`, `azure`, or `http` for URL construction of the corresponding
common backend:

```bash
cargo add lerobot-dataset-io --features aws
```

Then construct a source with `DatasetSource::from_object_store_url`, supplying
the backend options/credentials accepted by `object_store::parse_url_opts`.
The stored source identity excludes URL userinfo, query parameters, and
fragments so credentials are not copied into logs or manifests. Prefer backend
options or environment-based credential providers over credentials in a URL.
The `DatasetFileProvider` trait remains independently implementable for
backends outside that ecosystem.

Remote metadata is materialized eagerly. Packed data and video files are
materialized on demand through the async ensure and prefetch methods before
synchronous Parquet or media operations.

## Video export boundary

The crate plans video segments but deliberately does not choose or bundle a
media framework. Re-encoding exports receive a caller-provided
`VideoSegmentWriter`. Use `export_subset_preserving_packed_videos` when selected
packed video files should be copied byte-for-byte instead.

Every synchronous export requires `PreparedExportInputs`. For local datasets,
construct it with `PreparedExportInputs::for_local_dataset`; for remote
datasets, await `Dataset::prepare_export_inputs`. The value is bound to the
exact dataset, ordered episode selection, and video keys, and is verified
before the destination is touched.

## Format extensions

Unknown `info.json` fields survive subset export. Applications that interpret
their own namespaced fields or sidecars can implement `DatasetExportExtension`
and use the extension-aware export functions. Product-specific metadata and
cloud authentication belong in adapter crates rather than this crate.

## License

Apache-2.0.
