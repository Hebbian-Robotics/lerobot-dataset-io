use std::collections::BTreeMap;
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use crate::{
    FramesPerSecond, InvalidFramesPerSecond, PathTemplateRenderError, render_path_template,
};

/// Failure to parse a packed `LeRobot` dataset's `meta/info.json` document.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DatasetInfoParseError {
    /// The document is malformed JSON or does not contain the required fields.
    #[error("JSON error in info.json: {source}")]
    Json {
        /// The underlying JSON parser failure.
        #[source]
        source: serde_json::Error,
    },

    /// The document declares a `LeRobot` version outside the supported v3 family.
    #[error("unsupported LeRobot codebase version {found:?} (this reader supports v3.x)")]
    UnsupportedVersion {
        /// The unsupported version declared by the dataset.
        found: String,
    },

    /// The document declares a frame rate that cannot describe timed frames.
    #[error("invalid fps in info.json: {source}")]
    InvalidFramesPerSecond {
        /// Rejected frame-rate value.
        #[source]
        source: InvalidFramesPerSecond,
    },

    /// A data or video path template cannot produce a safe relative path.
    #[error("invalid {field} in info.json: {source}")]
    InvalidPathTemplate {
        /// JSON field containing the invalid template.
        field: &'static str,
        /// Template syntax or rendered-path failure.
        #[source]
        source: PathTemplateRenderError,
    },
}

/// Semantic role of a feature whose `LeRobot` dtype is `video`.
///
/// `LeRobot` uses the same dtype for RGB imagery and encoded depth maps. The
/// distinction matters to downstream consumers: depth video is useful raw
/// sensor data, but it must not be sent to an RGB image embedder by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoFeatureKind {
    /// Color video suitable for RGB-oriented consumers.
    Rgb,
    /// Encoded depth-map video.
    Depth,
    /// A video feature without enough metadata to classify safely.
    Unknown,
}

/// Data type of a feature declared in `meta/info.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureDtype {
    /// Encoded video referenced by packed-video metadata.
    Video,
    /// Image-valued frame data.
    Image,
    /// 32-bit floating-point values.
    Float32,
    /// 64-bit floating-point values.
    Float64,
    /// Signed 16-bit integer values.
    Int16,
    /// Signed 32-bit integer values.
    Int32,
    /// Signed 64-bit integer values.
    Int64,
    /// Unsigned 8-bit integer values.
    UInt8,
    /// Boolean values.
    Bool,
    /// UTF-8 string values.
    String,
    /// Forward-compatible catch-all preserving the raw dtype string.
    Unknown(String),
}

impl FeatureDtype {
    fn parse(raw_dtype: &str) -> Self {
        match raw_dtype {
            "video" => Self::Video,
            "image" => Self::Image,
            "float32" => Self::Float32,
            "float64" => Self::Float64,
            "int16" => Self::Int16,
            "int32" => Self::Int32,
            "int64" => Self::Int64,
            "uint8" => Self::UInt8,
            "bool" => Self::Bool,
            "string" => Self::String,
            unknown_dtype => {
                tracing::warn!(
                    dtype = unknown_dtype,
                    "unknown feature dtype in info.json; preserving raw value"
                );
                Self::Unknown(unknown_dtype.to_owned())
            }
        }
    }
}

/// One feature entry from the `features` object of `meta/info.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSpec {
    dtype: FeatureDtype,
    /// The feature's complete raw JSON object, preserved for round-trip export.
    raw: Value,
}

impl FeatureSpec {
    /// Parsed feature data type, including an explicit unknown variant.
    #[must_use]
    pub fn dtype(&self) -> &FeatureDtype {
        &self.dtype
    }

    /// Complete raw feature object retained for lossless round trips.
    #[must_use]
    pub fn raw(&self) -> &Value {
        &self.raw
    }

    /// Classifies a video feature as RGB, depth, or unknown.
    ///
    /// Published `LeRobot` datasets currently use both `info.is_depth_map` and
    /// `info["video.is_depth_map"]`; explicit metadata wins. Name and shape
    /// fallbacks cover older datasets: the channel axis is located through the
    /// feature's `names` axis list when one names it (channel-first layouts
    /// exist in the wild), defaulting to the last axis otherwise. Ambiguous
    /// one-channel imagery remains unknown instead of being silently treated
    /// as RGB.
    #[must_use]
    pub fn video_kind(&self, feature_name: &str) -> Option<VideoFeatureKind> {
        if self.dtype != FeatureDtype::Video {
            return None;
        }

        let explicit_depth_marker = self
            .raw
            .get("info")
            .and_then(Value::as_object)
            .and_then(|feature_info| {
                feature_info
                    .get("is_depth_map")
                    .or_else(|| feature_info.get("video.is_depth_map"))
            })
            .and_then(Value::as_bool);
        if let Some(is_depth_map) = explicit_depth_marker {
            return Some(if is_depth_map {
                VideoFeatureKind::Depth
            } else {
                VideoFeatureKind::Rgb
            });
        }

        let normalized_feature_name = feature_name.to_ascii_lowercase();
        if normalized_feature_name.contains("depth") {
            return Some(VideoFeatureKind::Depth);
        }
        if normalized_feature_name.contains("rgb") || normalized_feature_name.contains(".images.") {
            return Some(VideoFeatureKind::Rgb);
        }

        let named_channel_axis_position =
            self.raw
                .get("names")
                .and_then(Value::as_array)
                .and_then(|axis_names| {
                    axis_names.iter().position(|axis_name| {
                        axis_name.as_str().is_some_and(|axis_name| {
                            axis_name.eq_ignore_ascii_case("channel")
                                || axis_name.eq_ignore_ascii_case("channels")
                        })
                    })
                });
        let channel_count = self
            .raw
            .get("shape")
            .and_then(Value::as_array)
            .and_then(|shape| match named_channel_axis_position {
                Some(axis_position) => shape.get(axis_position),
                None => shape.last(),
            })
            .and_then(Value::as_u64);
        Some(if channel_count == Some(3) {
            VideoFeatureKind::Rgb
        } else {
            VideoFeatureKind::Unknown
        })
    }
}

/// Parsed `meta/info.json` of a `LeRobot` v3 dataset.
///
/// Unknown top-level fields are tolerated and preserved in the raw document so
/// that export can round-trip them.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    codebase_version: String,
    robot_type: Option<String>,
    frames_per_second: FramesPerSecond,
    total_episodes: u32,
    total_frames: u64,
    total_tasks: Option<u64>,
    splits: BTreeMap<String, String>,
    /// Template like `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`.
    data_path_template: String,
    /// Template like `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`.
    video_path_template: String,
    /// Features in `info.json` document order; this order defines the order of
    /// [`DatasetInfo::video_keys`].
    features: Vec<(String, FeatureSpec)>,
    /// The complete raw document, kept so export can preserve fields this
    /// reader does not model.
    raw_document: Value,
}

impl DatasetInfo {
    /// Parses the text of a `meta/info.json` document.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetInfoParseError::Json`] for malformed or structurally
    /// invalid JSON, and [`DatasetInfoParseError::UnsupportedVersion`] when
    /// `codebase_version` is not a v3.x version.
    pub fn parse_json(info_json_text: &str) -> Result<Self, DatasetInfoParseError> {
        let raw_document: Value = serde_json::from_str(info_json_text)
            .map_err(|source| DatasetInfoParseError::Json { source })?;
        let document: InfoDocument = serde_json::from_str(info_json_text)
            .map_err(|source| DatasetInfoParseError::Json { source })?;

        if !document.codebase_version.starts_with("v3.") {
            return Err(DatasetInfoParseError::UnsupportedVersion {
                found: document.codebase_version,
            });
        }
        if document.codebase_version != "v3.0" {
            tracing::warn!(
                codebase_version = document.codebase_version,
                "dataset uses a v3.x version newer than v3.0; proceeding optimistically"
            );
        }
        if document.features.0.is_empty() {
            tracing::warn!("info.json declares no features; dataset has no video keys");
        }

        let frames_per_second = FramesPerSecond::new(document.fps)
            .map_err(|source| DatasetInfoParseError::InvalidFramesPerSecond { source })?;
        render_path_template(&document.data_path, None, 0, 0).map_err(|source| {
            DatasetInfoParseError::InvalidPathTemplate {
                field: "data_path",
                source,
            }
        })?;
        render_path_template(&document.video_path, Some("video-key"), 0, 0).map_err(|source| {
            DatasetInfoParseError::InvalidPathTemplate {
                field: "video_path",
                source,
            }
        })?;

        let features: Vec<(String, FeatureSpec)> = document
            .features
            .0
            .into_iter()
            .map(|(feature_name, raw_feature)| {
                let dtype = raw_feature
                    .get("dtype")
                    .and_then(Value::as_str)
                    .map_or_else(
                        || {
                            tracing::warn!(
                                feature = feature_name,
                                "feature is missing a dtype; treating as unknown"
                            );
                            FeatureDtype::Unknown(String::new())
                        },
                        FeatureDtype::parse,
                    );
                (
                    feature_name,
                    FeatureSpec {
                        dtype,
                        raw: raw_feature,
                    },
                )
            })
            .collect();

        warn_on_unclassified_video_features(&features);

        Ok(Self {
            codebase_version: document.codebase_version,
            robot_type: document.robot_type,
            frames_per_second,
            total_episodes: document.total_episodes,
            total_frames: document.total_frames,
            total_tasks: document.total_tasks,
            splits: document.splits,
            data_path_template: document.data_path,
            video_path_template: document.video_path,
            features,
            raw_document,
        })
    }

    /// Declared `LeRobot` codebase version.
    #[must_use]
    pub fn codebase_version(&self) -> &str {
        &self.codebase_version
    }

    /// Declared robot type, when present.
    #[must_use]
    pub fn robot_type(&self) -> Option<&str> {
        self.robot_type.as_deref()
    }

    /// Validated dataset frame rate.
    #[must_use]
    pub const fn frames_per_second(&self) -> FramesPerSecond {
        self.frames_per_second
    }

    /// Number of episodes declared by `info.json`.
    #[must_use]
    pub const fn total_episodes(&self) -> u32 {
        self.total_episodes
    }

    /// Total frame count declared by `info.json`.
    #[must_use]
    pub const fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Number of tasks declared by `info.json`, when present.
    #[must_use]
    pub const fn total_tasks(&self) -> Option<u64> {
        self.total_tasks
    }

    /// Named dataset splits declared by `info.json`.
    #[must_use]
    pub fn splits(&self) -> &BTreeMap<String, String> {
        &self.splits
    }

    /// Parsed feature declarations in source-document order.
    #[must_use]
    pub fn features(&self) -> &[(String, FeatureSpec)] {
        &self.features
    }

    pub(crate) fn data_path_template(&self) -> &str {
        &self.data_path_template
    }

    pub(crate) fn video_path_template(&self) -> &str {
        &self.video_path_template
    }

    /// Camera keys, i.e. names of features with `dtype == "video"`, in
    /// `info.json` document order.
    #[must_use]
    pub fn video_keys(&self) -> Vec<String> {
        self.features
            .iter()
            .filter(|(_, spec)| spec.dtype == FeatureDtype::Video)
            .map(|(feature_name, _)| feature_name.clone())
            .collect()
    }

    /// RGB camera keys in `info.json` document order.
    ///
    /// Unlike [`Self::video_keys`], this deliberately excludes encoded depth
    /// maps and unclassified video features.
    #[must_use]
    pub fn rgb_video_keys(&self) -> Vec<String> {
        self.video_keys_of_kind(VideoFeatureKind::Rgb)
    }

    /// Encoded depth-video keys in `info.json` document order.
    #[must_use]
    pub fn depth_video_keys(&self) -> Vec<String> {
        self.video_keys_of_kind(VideoFeatureKind::Depth)
    }

    /// Video keys whose modality cannot be classified safely.
    #[must_use]
    pub fn unknown_video_keys(&self) -> Vec<String> {
        self.video_keys_of_kind(VideoFeatureKind::Unknown)
    }

    /// Returns the complete source document, including unmodeled fields.
    ///
    /// This supports lossless subset exports and format extensions without
    /// making extension-specific metadata part of the core model.
    #[must_use]
    pub fn raw_document(&self) -> &Value {
        &self.raw_document
    }

    fn video_keys_of_kind(&self, requested_kind: VideoFeatureKind) -> Vec<String> {
        self.features
            .iter()
            .filter(|(feature_name, feature_spec)| {
                feature_spec.video_kind(feature_name) == Some(requested_kind)
            })
            .map(|(feature_name, _)| feature_name.clone())
            .collect()
    }
}

/// Unclassified video features silently vanish from the embeddable,
/// indexable, and re-encodable camera surface downstream, so their exclusion
/// must be loud at the parse boundary.
fn warn_on_unclassified_video_features(features: &[(String, FeatureSpec)]) {
    let unclassified_video_keys: Vec<&String> = features
        .iter()
        .filter(|(feature_name, feature_spec)| {
            feature_spec.video_kind(feature_name) == Some(VideoFeatureKind::Unknown)
        })
        .map(|(feature_name, _)| feature_name)
        .collect();
    if !unclassified_video_keys.is_empty() {
        tracing::warn!(
            video_keys = ?unclassified_video_keys,
            "video features could not be classified as RGB or depth; \
             excluded from embedding, indexing, and re-encoding exports"
        );
    }
}

#[derive(Deserialize)]
struct InfoDocument {
    codebase_version: String,
    robot_type: Option<String>,
    total_episodes: u32,
    total_frames: u64,
    total_tasks: Option<u64>,
    fps: f64,
    #[serde(default)]
    splits: BTreeMap<String, String>,
    data_path: String,
    video_path: String,
    #[serde(default)]
    features: OrderedFeatures,
}

/// The `features` object with entries kept in document order.
///
/// `serde_json::Map` is backed by a `BTreeMap` (alphabetical), but the order
/// of camera features in the document defines `video_keys` order, so this
/// custom visitor captures entries as they appear.
#[derive(Debug, Clone, Default)]
struct OrderedFeatures(Vec<(String, Value)>);

impl<'de> Deserialize<'de> for OrderedFeatures {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedFeaturesVisitor;

        impl<'de> Visitor<'de> for OrderedFeaturesVisitor {
            type Value = OrderedFeatures;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a JSON object mapping feature names to feature specs")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some((feature_name, raw_feature)) = map.next_entry::<String, Value>()? {
                    entries.push((feature_name, raw_feature));
                }
                Ok(OrderedFeatures(entries))
            }
        }

        deserializer.deserialize_map(OrderedFeaturesVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_V3_INFO_JSON: &str = r#"{
        "codebase_version": "v3.0",
        "robot_type": "so100_follower",
        "total_episodes": 50,
        "total_frames": 11939,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": { "train": "0:50" },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": { "dtype": "float32", "shape": [6], "names": null, "fps": 30.0 },
            "observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] },
            "observation.images.side": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] },
            "timestamp": { "dtype": "float32", "shape": [1], "names": null }
        },
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500
    }"#;

    #[test]
    fn parses_minimal_v3_info() {
        let info = DatasetInfo::parse_json(MINIMAL_V3_INFO_JSON).unwrap();
        assert_eq!(info.codebase_version, "v3.0");
        assert_eq!(info.robot_type.as_deref(), Some("so100_follower"));
        assert_eq!(info.total_episodes, 50);
        assert_eq!(info.total_frames, 11939);
        assert_eq!(info.total_tasks, Some(1));
        assert!((info.frames_per_second().get() - 30.0).abs() < f64::EPSILON);
        assert_eq!(info.splits.get("train").map(String::as_str), Some("0:50"));
        assert_eq!(
            info.data_path_template,
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        );
        assert_eq!(
            info.video_path_template,
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        );
        assert_eq!(info.features.len(), 4);
    }

    #[test]
    fn video_keys_preserve_document_order() {
        // "up" precedes "side" in the document even though alphabetical order
        // would reverse them; document order must win.
        let info = DatasetInfo::parse_json(MINIMAL_V3_INFO_JSON).unwrap();
        assert_eq!(
            info.video_keys(),
            vec![
                "observation.images.up".to_owned(),
                "observation.images.side".to_owned()
            ]
        );
    }

    #[test]
    fn classifies_rgb_and_depth_video_from_published_metadata_variants() {
        let text = MINIMAL_V3_INFO_JSON.replacen(
            r#""observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
            r#""observation.depth.up": { "dtype": "video", "shape": [480, 640, 1], "names": ["height", "width", "channels"], "info": { "is_depth_map": true } }"#,
            1,
        );
        let info = DatasetInfo::parse_json(&text).unwrap();

        assert_eq!(
            info.rgb_video_keys(),
            vec!["observation.images.side".to_owned()]
        );
        assert_eq!(
            info.depth_video_keys(),
            vec!["observation.depth.up".to_owned()]
        );
        assert!(info.unknown_video_keys().is_empty());
    }

    #[test]
    fn explicit_non_depth_marker_classifies_rgb_without_name_heuristics() {
        let text = MINIMAL_V3_INFO_JSON.replacen(
            r#""observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
            r#""camera_0": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"], "info": { "video.is_depth_map": false } }"#,
            1,
        );
        let info = DatasetInfo::parse_json(&text).unwrap();

        assert_eq!(
            info.rgb_video_keys(),
            vec!["camera_0".to_owned(), "observation.images.side".to_owned()]
        );
    }

    #[test]
    fn ambiguous_video_feature_is_not_assumed_to_be_rgb() {
        let text = MINIMAL_V3_INFO_JSON.replacen(
            r#""observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
            r#""camera_0": { "dtype": "video", "shape": [480, 640, 1], "names": ["height", "width", "channels"] }"#,
            1,
        );
        let info = DatasetInfo::parse_json(&text).unwrap();

        assert_eq!(info.unknown_video_keys(), vec!["camera_0".to_owned()]);
    }

    #[test]
    fn channel_first_rgb_video_is_classified_via_names_axis() {
        // The feature name deliberately misses every name heuristic and the
        // channel axis is FIRST, so only the `names` axis list can classify it.
        let text = MINIMAL_V3_INFO_JSON.replacen(
            r#""observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
            r#""cam_high": { "dtype": "video", "shape": [3, 480, 640], "names": ["channels", "height", "width"] }"#,
            1,
        );
        let info = DatasetInfo::parse_json(&text).unwrap();

        assert_eq!(
            info.rgb_video_keys(),
            vec!["cam_high".to_owned(), "observation.images.side".to_owned()]
        );
        assert!(info.unknown_video_keys().is_empty());
    }

    #[test]
    fn channel_first_single_channel_video_stays_unknown() {
        // One channel could be depth or grayscale; without an explicit depth
        // marker it must stay unclassified rather than be guessed.
        let text = MINIMAL_V3_INFO_JSON.replacen(
            r#""observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
            r#""cam_high": { "dtype": "video", "shape": [1, 480, 640], "names": ["channels", "height", "width"] }"#,
            1,
        );
        let info = DatasetInfo::parse_json(&text).unwrap();

        assert_eq!(info.unknown_video_keys(), vec!["cam_high".to_owned()]);
    }

    #[test]
    fn names_lacking_a_channel_axis_fall_back_to_last_axis_classification() {
        // Axis lists without a channel entry and non-array `names` (LeRobot
        // also publishes objects like {"motors": [...]}) both keep the
        // last-axis convention.
        let text = MINIMAL_V3_INFO_JSON
            .replacen(
                r#""observation.images.up": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
                r#""cam_high": { "dtype": "video", "shape": [480, 640, 3], "names": ["dim0", "dim1", "dim2"] }"#,
                1,
            )
            .replacen(
                r#""observation.images.side": { "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"] }"#,
                r#""cam_low": { "dtype": "video", "shape": [480, 640, 3], "names": { "axes": ["channels", "height", "width"] } }"#,
                1,
            );
        let info = DatasetInfo::parse_json(&text).unwrap();

        assert_eq!(
            info.rgb_video_keys(),
            vec!["cam_high".to_owned(), "cam_low".to_owned()]
        );
    }

    #[test]
    fn tolerates_unknown_top_level_fields_and_preserves_them() {
        let text = MINIMAL_V3_INFO_JSON.replacen(
            "\"codebase_version\"",
            "\"some_future_field\": {\"nested\": [1, 2]}, \"codebase_version\"",
            1,
        );
        let info = DatasetInfo::parse_json(&text).unwrap();
        assert_eq!(info.total_episodes, 50);
        assert!(info.raw_document.get("some_future_field").is_some());
    }

    #[test]
    fn unknown_feature_dtype_parses_to_unknown_variant() {
        let text = MINIMAL_V3_INFO_JSON.replacen("\"float32\"", "\"quaternion9000\"", 1);
        let info = DatasetInfo::parse_json(&text).unwrap();
        let (_, action_spec) = info
            .features
            .iter()
            .find(|(feature_name, _)| feature_name == "action")
            .unwrap();
        assert_eq!(
            action_spec.dtype,
            FeatureDtype::Unknown("quaternion9000".to_owned())
        );
        // Unknown dtypes must not surface as video keys.
        assert_eq!(info.video_keys().len(), 2);
    }

    #[test]
    fn feature_missing_dtype_parses_to_unknown_variant() {
        let text = MINIMAL_V3_INFO_JSON.replacen("\"dtype\": \"float32\", ", "", 1);
        let info = DatasetInfo::parse_json(&text).unwrap();
        let (_, action_spec) = info
            .features
            .iter()
            .find(|(feature_name, _)| feature_name == "action")
            .unwrap();
        assert!(matches!(action_spec.dtype, FeatureDtype::Unknown(_)));
    }

    #[test]
    fn v2_version_is_rejected() {
        let text = MINIMAL_V3_INFO_JSON.replacen("v3.0", "v2.1", 1);
        let error = DatasetInfo::parse_json(&text).unwrap_err();
        assert!(
            matches!(error, DatasetInfoParseError::UnsupportedVersion { ref found } if found == "v2.1"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn newer_v3_minor_version_is_accepted() {
        let text = MINIMAL_V3_INFO_JSON.replacen("v3.0", "v3.1", 1);
        let info = DatasetInfo::parse_json(&text).unwrap();
        assert_eq!(info.codebase_version, "v3.1");
    }

    #[test]
    fn malformed_json_is_a_json_error() {
        let error = DatasetInfo::parse_json("{ not json").unwrap_err();
        assert!(matches!(error, DatasetInfoParseError::Json { .. }));
    }

    #[test]
    fn missing_required_field_is_a_json_error() {
        let text = MINIMAL_V3_INFO_JSON.replacen("\"total_frames\": 11939,", "", 1);
        let error = DatasetInfo::parse_json(&text).unwrap_err();
        assert!(matches!(error, DatasetInfoParseError::Json { .. }));
    }

    #[test]
    fn missing_features_yields_empty_video_keys() {
        let start = MINIMAL_V3_INFO_JSON.find("\"features\"").unwrap();
        let end = MINIMAL_V3_INFO_JSON
            .find("\"data_files_size_in_mb\"")
            .unwrap();
        let mut text = String::new();
        text.push_str(&MINIMAL_V3_INFO_JSON[..start]);
        text.push_str(&MINIMAL_V3_INFO_JSON[end..]);
        let info = DatasetInfo::parse_json(&text).unwrap();
        assert!(info.video_keys().is_empty());
    }

    #[test]
    fn integer_fps_parses_as_float() {
        let info = DatasetInfo::parse_json(MINIMAL_V3_INFO_JSON).unwrap();
        assert!((info.frames_per_second().get() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rejects_invalid_frame_rates_and_path_templates_at_the_metadata_boundary() {
        for invalid_fps in ["0", "-1"] {
            let invalid_document =
                MINIMAL_V3_INFO_JSON.replacen("\"fps\": 30", &format!("\"fps\": {invalid_fps}"), 1);
            assert!(matches!(
                DatasetInfo::parse_json(&invalid_document),
                Err(DatasetInfoParseError::InvalidFramesPerSecond { .. })
            ));
        }

        for (field, valid_template, invalid_template) in [
            (
                "data_path",
                "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "../outside-{file_index}.parquet",
            ),
            (
                "video_path",
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "/absolute/{video_key}.mp4",
            ),
        ] {
            let invalid_document =
                MINIMAL_V3_INFO_JSON.replacen(valid_template, invalid_template, 1);
            assert!(matches!(
                DatasetInfo::parse_json(&invalid_document),
                Err(DatasetInfoParseError::InvalidPathTemplate {
                    field: rejected_field,
                    ..
                }) if rejected_field == field
            ));
        }
    }
}
