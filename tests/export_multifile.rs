//! Synthetic (fixture-free) export test: a dataset whose two data parquet
//! files share the same fields in DIFFERENT column orders must export with
//! every value mapped by column name, not by position.

use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use lerobot_dataset_io::{
    CameraVideoWriteResult, Dataset, SourceSegment, VideoSegmentWriter, export_subset,
};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// The synthetic dataset has no cameras, so the video writer must never run.
struct NoVideosWriter;

impl VideoSegmentWriter for NoVideosWriter {
    fn write_camera_video(
        &self,
        camera: &str,
        _segments: &[SourceSegment],
        _dest_mp4: &Path,
    ) -> Result<CameraVideoWriteResult, Box<dyn std::error::Error + Send + Sync>> {
        panic!("synthetic dataset has no cameras; unexpected call for {camera}");
    }
}

fn int64_column(name: &str, values: Vec<i64>) -> (&str, ArrayRef) {
    (name, Arc::new(Int64Array::from(values)) as ArrayRef)
}

fn write_parquet(path: &Path, columns: Vec<(&str, ArrayRef)>) {
    let batch = RecordBatch::try_from_iter(columns).expect("valid synthetic batch");
    fs::create_dir_all(path.parent().expect("parquet paths have parents")).unwrap();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Builds a two-episode v3 dataset where episode 0 lives in
/// `data/chunk-000/file-000.parquet` and episode 1 in `file-001.parquet`,
/// with the SAME fields in a different column order.
fn build_multifile_dataset(root: &Path) {
    fs::create_dir_all(root.join("meta")).unwrap();
    fs::write(
        root.join("meta").join("info.json"),
        r#"{
            "codebase_version": "v3.0",
            "robot_type": null,
            "total_episodes": 2,
            "total_frames": 5,
            "total_tasks": 2,
            "fps": 30,
            "splits": { "train": "0:2" },
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {}
        }"#,
    )
    .unwrap();

    let mut tasks_builder = ListBuilder::new(StringBuilder::new());
    tasks_builder.values().append_value("solder");
    tasks_builder.append(true);
    tasks_builder.values().append_value("inspect");
    tasks_builder.append(true);
    write_parquet(
        &root
            .join("meta")
            .join("episodes")
            .join("chunk-000")
            .join("file-000.parquet"),
        vec![
            int64_column("episode_index", vec![0, 1]),
            ("tasks", Arc::new(tasks_builder.finish()) as ArrayRef),
            int64_column("length", vec![3, 2]),
            int64_column("dataset_from_index", vec![0, 3]),
            int64_column("dataset_to_index", vec![3, 5]),
            int64_column("data/chunk_index", vec![0, 0]),
            int64_column("data/file_index", vec![0, 1]),
        ],
    );

    write_parquet(
        &root.join("meta").join("tasks.parquet"),
        vec![
            (
                "task",
                Arc::new(StringArray::from(vec!["solder", "inspect"])) as ArrayRef,
            ),
            int64_column("task_index", vec![0, 1]),
        ],
    );

    // Episode 0: bookkeeping columns FIRST.
    write_parquet(
        &root.join("data").join("chunk-000").join("file-000.parquet"),
        vec![
            int64_column("episode_index", vec![0, 0, 0]),
            int64_column("frame_index", vec![0, 1, 2]),
            int64_column("index", vec![0, 1, 2]),
            int64_column("task_index", vec![0, 0, 0]),
            int64_column("value", vec![100, 101, 102]),
        ],
    );
    // Episode 1: the SAME fields, deliberately reordered.
    write_parquet(
        &root.join("data").join("chunk-000").join("file-001.parquet"),
        vec![
            int64_column("value", vec![200, 201]),
            int64_column("task_index", vec![1, 1]),
            int64_column("frame_index", vec![0, 1]),
            int64_column("episode_index", vec![1, 1]),
            int64_column("index", vec![3, 4]),
        ],
    );
}

fn read_int64_columns(path: &Path, column_names: &[&str]) -> Vec<Vec<i64>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
        .unwrap()
        .build()
        .unwrap();
    let mut columns: Vec<Vec<i64>> = vec![Vec::new(); column_names.len()];
    for batch_result in reader {
        let batch = batch_result.unwrap();
        for (position, column_name) in column_names.iter().enumerate() {
            let values = batch
                .column_by_name(column_name)
                .unwrap_or_else(|| panic!("missing column {column_name}"))
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap_or_else(|| panic!("column {column_name} is not int64"));
            columns[position].extend(values.values());
        }
    }
    columns
}

#[test]
fn export_maps_columns_by_name_across_data_files_with_different_field_orders() {
    let workspace = tempfile::tempdir().unwrap();
    let source_root = workspace.path().join("source");
    build_multifile_dataset(&source_root);
    let source = Dataset::open(&source_root).unwrap();

    // Episode 1 comes FIRST, so the exported schema mirrors the reordered
    // file-001 layout and episode 0's batches must be remapped by name.
    let dest_root = workspace.path().join("exported");
    let export_episodes = lerobot_dataset_io::ExportEpisodes::try_from(vec![1, 0])
        .expect("non-empty export episodes");
    let prepared_inputs = lerobot_dataset_io::PreparedExportInputs::for_local_dataset(
        &source,
        &export_episodes,
        source.video_keys(),
    )
    .unwrap();
    let report = export_subset(
        &source,
        &export_episodes,
        &dest_root,
        &NoVideosWriter,
        prepared_inputs,
    )
    .unwrap();
    assert_eq!(report.episodes_written, 2);
    assert_eq!(report.frames_written, 5);

    let exported_data_path = dest_root
        .join("data")
        .join("chunk-000")
        .join("file-000.parquet");
    let [
        values,
        episode_indices,
        global_indices,
        frame_indices,
        task_indices,
    ] = <[Vec<i64>; 5]>::try_from(read_int64_columns(
        &exported_data_path,
        &[
            "value",
            "episode_index",
            "index",
            "frame_index",
            "task_index",
        ],
    ))
    .unwrap();

    // Payload values must follow their column NAME, never their position.
    assert_eq!(values, vec![200, 201, 100, 101, 102]);
    assert_eq!(episode_indices, vec![0, 0, 1, 1, 1]);
    assert_eq!(global_indices, vec![0, 1, 2, 3, 4]);
    assert_eq!(frame_indices, vec![0, 1, 0, 1, 2]);
    // Tasks remap in first-appearance order: "inspect" -> 0, "solder" -> 1.
    assert_eq!(task_indices, vec![0, 0, 1, 1, 1]);

    let exported = Dataset::open(&dest_root).unwrap();
    assert_eq!(exported.episodes().len(), 2);
    assert_eq!(exported.episodes()[0].tasks(), ["inspect".to_owned()]);
    assert_eq!(exported.episodes()[1].tasks(), ["solder".to_owned()]);
    assert_eq!(exported.episodes()[0].length_frames(), 2);
    assert_eq!(exported.episodes()[1].length_frames(), 3);
}

#[test]
fn prepared_inputs_cannot_authorize_a_different_export_request() {
    let workspace = tempfile::tempdir().unwrap();
    let source_root = workspace.path().join("source");
    build_multifile_dataset(&source_root);
    let source = Dataset::open(&source_root).unwrap();
    let prepared_episode = lerobot_dataset_io::ExportEpisodes::try_from(vec![0]).unwrap();
    let requested_episode = lerobot_dataset_io::ExportEpisodes::try_from(vec![1]).unwrap();
    let prepared_inputs = lerobot_dataset_io::PreparedExportInputs::for_local_dataset(
        &source,
        &prepared_episode,
        source.video_keys(),
    )
    .unwrap();
    let destination_root = workspace.path().join("must-not-be-created");

    let error = export_subset(
        &source,
        &requested_episode,
        &destination_root,
        &NoVideosWriter,
        prepared_inputs,
    )
    .expect_err("request mismatch must be rejected");

    assert!(matches!(
        error,
        lerobot_dataset_io::LerobotError::Source { .. }
    ));
    assert!(!destination_root.exists());
}
