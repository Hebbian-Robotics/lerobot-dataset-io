//! Typed accessors for arrow columns addressed by their exact (slash- and
//! dot-containing) `LeRobot` column names.

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::Schema;

use crate::LerobotError;

pub fn require_column<'a>(
    batch: &'a RecordBatch,
    column_name: &str,
) -> Result<&'a ArrayRef, LerobotError> {
    batch
        .column_by_name(column_name)
        .ok_or_else(|| LerobotError::MissingColumn {
            name: column_name.to_owned(),
        })
}

/// Returns a required non-nullable `Int64` column by its exact `LeRobot` name.
///
/// # Errors
///
/// Returns [`LerobotError`] when the column is missing or has another type.
pub fn require_int64_column<'a>(
    batch: &'a RecordBatch,
    column_name: &str,
) -> Result<&'a Int64Array, LerobotError> {
    require_column(batch, column_name)?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| type_mismatch(column_name, "int64"))
}

pub fn require_float64_column<'a>(
    batch: &'a RecordBatch,
    column_name: &str,
) -> Result<&'a Float64Array, LerobotError> {
    require_column(batch, column_name)?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| type_mismatch(column_name, "float64"))
}

/// Returns a required UTF-8 column by its exact `LeRobot` name.
///
/// # Errors
///
/// Returns [`LerobotError`] when the column is missing or has another type.
pub fn require_string_column<'a>(
    batch: &'a RecordBatch,
    column_name: &str,
) -> Result<&'a StringArray, LerobotError> {
    require_column(batch, column_name)?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| type_mismatch(column_name, "utf8"))
}

pub fn require_list_column<'a>(
    batch: &'a RecordBatch,
    column_name: &str,
) -> Result<&'a ListArray, LerobotError> {
    require_column(batch, column_name)?
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| type_mismatch(column_name, "list"))
}

pub fn require_field_position(schema: &Schema, column_name: &str) -> Result<usize, LerobotError> {
    schema
        .column_with_name(column_name)
        .map(|(position, _)| position)
        .ok_or_else(|| LerobotError::MissingColumn {
            name: column_name.to_owned(),
        })
}

pub fn int64_row_value(
    column: &Int64Array,
    row: usize,
    column_name: &str,
) -> Result<i64, LerobotError> {
    if column.is_null(row) {
        return Err(null_value(column_name, row));
    }
    Ok(column.value(row))
}

pub fn float64_row_value(
    column: &Float64Array,
    row: usize,
    column_name: &str,
) -> Result<f64, LerobotError> {
    if column.is_null(row) {
        return Err(null_value(column_name, row));
    }
    Ok(column.value(row))
}

/// Whether `column_name` holds a null at `row`.
///
/// A MISSING column is malformed metadata and still errors; a null VALUE is a
/// fact a dataset is entitled to state, and callers that treat it as one need
/// to ask before reading. Every typed accessor here rejects nulls, which is the
/// right default — this is the deliberate opt-out.
pub fn row_is_null(
    batch: &RecordBatch,
    row: usize,
    column_name: &str,
) -> Result<bool, LerobotError> {
    Ok(require_column(batch, column_name)?.is_null(row))
}

pub fn i64_row(batch: &RecordBatch, row: usize, column_name: &str) -> Result<i64, LerobotError> {
    int64_row_value(require_int64_column(batch, column_name)?, row, column_name)
}

pub fn u32_row(batch: &RecordBatch, row: usize, column_name: &str) -> Result<u32, LerobotError> {
    let value = i64_row(batch, row, column_name)?;
    u32::try_from(value).map_err(|_| out_of_range(column_name, value, "u32"))
}

pub fn u64_row(batch: &RecordBatch, row: usize, column_name: &str) -> Result<u64, LerobotError> {
    let value = i64_row(batch, row, column_name)?;
    u64::try_from(value).map_err(|_| out_of_range(column_name, value, "u64"))
}

pub fn f64_row(batch: &RecordBatch, row: usize, column_name: &str) -> Result<f64, LerobotError> {
    float64_row_value(
        require_float64_column(batch, column_name)?,
        row,
        column_name,
    )
}

fn type_mismatch(column_name: &str, expected_type: &str) -> LerobotError {
    LerobotError::InconsistentMetadata {
        detail: format!("column {column_name:?} is not of expected type {expected_type}"),
    }
}

fn null_value(column_name: &str, row: usize) -> LerobotError {
    LerobotError::InconsistentMetadata {
        detail: format!("unexpected null in column {column_name:?} at row {row}"),
    }
}

fn out_of_range(column_name: &str, value: i64, target_type: &str) -> LerobotError {
    LerobotError::InconsistentMetadata {
        detail: format!("column {column_name:?} value {value} is out of range for {target_type}"),
    }
}
