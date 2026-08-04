//! A `LeRobot` v3 episode that has no footage for a declared camera can only
//! say so by nulling that camera's whole `videos/{key}/*` window group — the
//! schema is fixed per dataset, so it cannot omit the columns. These tests pin
//! that reading: absence is a fact the reader reports, not corruption it
//! refuses.

use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use lerobot_dataset_io::{Dataset, LerobotError};
use parquet::arrow::ArrowWriter;

const PRESENT_CAMERA: &str = "observation.rgb.top";
/// Recorded on the first episode and absent from the second.
const INTERMITTENT_CAMERA: &str = "observation.rgb.wrist";

/// One episode's window into a camera's packed file. `None` nulls the whole
/// group, which is how the dataset states "this episode has no such camera".
type EpisodeWindow = Option<CameraWindow>;

struct CameraWindow {
    chunk_index: i64,
    file_index: i64,
    from_timestamp: f64,
    to_timestamp: f64,
}

impl CameraWindow {
    fn at(from_timestamp: f64, to_timestamp: f64) -> Self {
        Self {
            chunk_index: 2,
            file_index: 7,
            from_timestamp,
            to_timestamp,
        }
    }
}

fn int64_column(name: impl Into<String>, values: Vec<i64>) -> (String, ArrayRef) {
    (name.into(), Arc::new(Int64Array::from(values)) as ArrayRef)
}

fn write_parquet(path: &Path, columns: Vec<(String, ArrayRef)>) {
    let batch = RecordBatch::try_from_iter(columns).expect("valid synthetic batch");
    fs::create_dir_all(path.parent().expect("parquet path has a parent")).unwrap();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Emits the four window columns for one camera across both episodes, nulling
/// on the LAST episode exactly the components named in `nulled_suffixes`
/// (empty for a well-formed group, all four for an absent camera, a subset for
/// a malformed one). The first episode always keeps its complete window, so
/// every case here is a within-dataset contrast.
fn window_columns(
    camera: &str,
    windows: [&EpisodeWindow; 2],
    nulled_suffixes: &[&str],
) -> Vec<(String, ArrayRef)> {
    let is_nulled = |episode_position: usize, suffix: &str| {
        episode_position + 1 == windows.len() && nulled_suffixes.contains(&suffix)
    };
    let int_values = |suffix: &str, component: fn(&CameraWindow) -> i64| {
        Arc::new(Int64Array::from(
            windows
                .iter()
                .enumerate()
                .map(|(episode_position, window)| {
                    window
                        .as_ref()
                        .filter(|_| !is_nulled(episode_position, suffix))
                        .map(&component)
                })
                .collect::<Vec<Option<i64>>>(),
        )) as ArrayRef
    };
    let float_values = |suffix: &str, component: fn(&CameraWindow) -> f64| {
        Arc::new(Float64Array::from(
            windows
                .iter()
                .enumerate()
                .map(|(episode_position, window)| {
                    window
                        .as_ref()
                        .filter(|_| !is_nulled(episode_position, suffix))
                        .map(&component)
                })
                .collect::<Vec<Option<f64>>>(),
        )) as ArrayRef
    };
    vec![
        (
            format!("videos/{camera}/chunk_index"),
            int_values("chunk_index", |window| window.chunk_index),
        ),
        (
            format!("videos/{camera}/file_index"),
            int_values("file_index", |window| window.file_index),
        ),
        (
            format!("videos/{camera}/from_timestamp"),
            float_values("from_timestamp", |window| window.from_timestamp),
        ),
        (
            format!("videos/{camera}/to_timestamp"),
            float_values("to_timestamp", |window| window.to_timestamp),
        ),
    ]
}

/// A two-episode, two-camera dataset. `PRESENT_CAMERA` is recorded on both
/// episodes; `INTERMITTENT_CAMERA` is recorded on the first only, with the
/// components in `nulled_suffixes` nulled on the second.
fn build_dataset(root: &Path, nulled_suffixes: &[&str]) {
    build_dataset_with_second_window(root, nulled_suffixes, CameraWindow::at(1.0, 2.0));
}

fn build_dataset_with_second_window(
    root: &Path,
    nulled_suffixes: &[&str],
    second_camera_window: CameraWindow,
) {
    fs::create_dir_all(root.join("meta")).unwrap();
    fs::write(
        root.join("meta/info.json"),
        format!(
            r#"{{
                "codebase_version": "v3.0",
                "robot_type": null,
                "total_episodes": 2,
                "total_frames": 4,
                "total_tasks": 1,
                "fps": 30,
                "splits": {{ "train": "0:2" }},
                "data_path": "data/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.parquet",
                "video_path": "videos/{{video_key}}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
                "features": {{
                    "{PRESENT_CAMERA}": {{
                        "dtype": "video",
                        "shape": [8, 8, 3],
                        "info": {{ "is_depth_map": false }}
                    }},
                    "{INTERMITTENT_CAMERA}": {{
                        "dtype": "video",
                        "shape": [8, 8, 3],
                        "info": {{ "is_depth_map": false }}
                    }}
                }}
            }}"#
        ),
    )
    .unwrap();

    let mut tasks_builder = ListBuilder::new(StringBuilder::new());
    for _ in 0..2 {
        tasks_builder.values().append_value("pick up the block");
        tasks_builder.append(true);
    }
    let mut episode_columns = vec![
        int64_column("episode_index", vec![0, 1]),
        (
            "tasks".to_owned(),
            Arc::new(tasks_builder.finish()) as ArrayRef,
        ),
        int64_column("length", vec![2, 2]),
        int64_column("dataset_from_index", vec![0, 2]),
        int64_column("dataset_to_index", vec![2, 4]),
        int64_column("data/chunk_index", vec![0, 0]),
        int64_column("data/file_index", vec![3, 3]),
    ];
    let first_window = Some(CameraWindow::at(0.0, 1.0));
    let second_window = Some(second_camera_window);
    episode_columns.extend(window_columns(
        PRESENT_CAMERA,
        [&first_window, &second_window],
        &[],
    ));
    episode_columns.extend(window_columns(
        INTERMITTENT_CAMERA,
        [&first_window, &second_window],
        nulled_suffixes,
    ));
    write_parquet(
        &root.join("meta/episodes/chunk-000/file-000.parquet"),
        episode_columns,
    );
}

/// Every window component, i.e. the whole group nulled: the encoding for
/// "this episode has no footage from this camera".
const ALL_WINDOW_COMPONENTS: [&str; 4] = [
    "chunk_index",
    "file_index",
    "from_timestamp",
    "to_timestamp",
];

#[test]
fn an_episode_that_nulls_a_camera_window_group_simply_lacks_that_camera() {
    let dataset_dir = tempfile::tempdir().unwrap();
    build_dataset(dataset_dir.path(), &ALL_WINDOW_COMPONENTS);

    let dataset = Dataset::open(dataset_dir.path()).expect("a partly-recorded camera opens fine");
    let episodes = dataset.episodes();

    assert_eq!(
        episodes[0].video_segments().keys().collect::<Vec<_>>(),
        vec![PRESENT_CAMERA, INTERMITTENT_CAMERA],
        "the episode that recorded both cameras keeps both segments"
    );
    assert_eq!(
        episodes[1].video_segments().keys().collect::<Vec<_>>(),
        vec![PRESENT_CAMERA],
        "the episode that nulled the wrist window has no wrist segment at all"
    );
    assert!(
        dataset
            .video_keys()
            .contains(&INTERMITTENT_CAMERA.to_owned()),
        "the camera stays DECLARED by the dataset; only this episode lacks footage"
    );
}

#[test]
fn a_partially_nulled_camera_window_group_is_rejected_as_malformed() {
    // A half-written row and a deliberately absent camera must not look the
    // same, or a truncating writer disappears as a silently dropped camera.
    for partial_group in [
        vec!["chunk_index"],
        vec!["from_timestamp", "to_timestamp"],
        vec!["chunk_index", "file_index", "from_timestamp"],
    ] {
        let dataset_dir = tempfile::tempdir().unwrap();
        build_dataset(dataset_dir.path(), &partial_group);

        let error = Dataset::open(dataset_dir.path())
            .expect_err(&format!("nulling only {partial_group:?} must be rejected"));

        let LerobotError::InconsistentMetadata { detail } = &error else {
            panic!("unexpected error for {partial_group:?}: {error}");
        };
        assert!(
            detail.contains(INTERMITTENT_CAMERA) && detail.contains("episode 1"),
            "the error must name the episode and camera it found: {detail}"
        );
    }
}

#[test]
fn a_fully_recorded_dataset_is_unaffected() {
    // The null-tolerant read must not change what a conventional dataset — the
    // overwhelming majority — parses to.
    let dataset_dir = tempfile::tempdir().unwrap();
    build_dataset(dataset_dir.path(), &[]);

    let dataset = Dataset::open(dataset_dir.path()).expect("a fully recorded dataset opens");

    for episode in dataset.episodes() {
        assert_eq!(
            episode.video_segments().keys().collect::<Vec<_>>(),
            vec![PRESENT_CAMERA, INTERMITTENT_CAMERA],
            "episode {} must keep every declared camera",
            episode.episode_index()
        );
    }
}

#[test]
fn an_invalid_video_timestamp_range_is_rejected_when_metadata_is_opened() {
    let dataset_dir = tempfile::tempdir().unwrap();
    build_dataset_with_second_window(dataset_dir.path(), &[], CameraWindow::at(2.0, 1.0));

    let error = Dataset::open(dataset_dir.path()).expect_err("reversed timestamp range");
    assert!(matches!(error, LerobotError::InconsistentMetadata { .. }));
}
