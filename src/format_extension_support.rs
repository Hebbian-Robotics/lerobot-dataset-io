//! Low-level readers shared with crates that implement `LeRobot` format extensions.
//!
//! Most consumers should use [`crate::Dataset`] and the top-level export API.
//! This module exists for extension adapters that own additional Parquet
//! sidecars while reusing the core crate's path-containment and Arrow-column
//! error semantics.

pub use crate::arrow_columns::{require_int64_column, require_string_column};
pub use crate::dataset::{open_parquet_reader_builder, safe_dataset_relative_path};
