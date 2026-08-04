use std::fmt;

/// A finite, strictly positive dataset frame rate.
///
/// `LeRobot` stores the value as a JSON number, but readers should not need to
/// defend against zero, negative, or non-finite rates after metadata parsing.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FramesPerSecond(f64);

impl FramesPerSecond {
    /// Parses a raw frame rate into the valid domain.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidFramesPerSecond`] when `value` is zero, negative, or
    /// non-finite.
    pub fn new(value: f64) -> Result<Self, InvalidFramesPerSecond> {
        if !value.is_finite() || value <= 0.0 {
            return Err(InvalidFramesPerSecond { value });
        }
        Ok(Self(value))
    }

    /// Returns the underlying frames-per-second value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }

    /// Converts a frame count to seconds.
    #[must_use]
    pub fn duration_for_frames(self, frame_count: u64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let frame_count = frame_count as f64;
        frame_count / self.0
    }
}

impl TryFrom<f64> for FramesPerSecond {
    type Error = InvalidFramesPerSecond;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<FramesPerSecond> for f64 {
    fn from(value: FramesPerSecond) -> Self {
        value.get()
    }
}

/// A raw value that cannot represent a dataset frame rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidFramesPerSecond {
    value: f64,
}

impl InvalidFramesPerSecond {
    /// Returns the rejected raw value.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }
}

impl fmt::Display for InvalidFramesPerSecond {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "frames per second must be finite and greater than zero, found {}",
            self.value
        )
    }
}

impl std::error::Error for InvalidFramesPerSecond {}

/// A finite, non-empty half-open timestamp range measured in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimestampRange {
    start_seconds: f64,
    end_seconds: f64,
}

impl TimestampRange {
    /// Parses an inclusive start and exclusive end into a valid range.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidTimestampRange`] unless both timestamps are finite,
    /// the start is non-negative, and the end is strictly after the start.
    pub fn new(start_seconds: f64, end_seconds: f64) -> Result<Self, InvalidTimestampRange> {
        if !start_seconds.is_finite()
            || !end_seconds.is_finite()
            || start_seconds < 0.0
            || end_seconds <= start_seconds
        {
            return Err(InvalidTimestampRange {
                start_seconds,
                end_seconds,
            });
        }
        Ok(Self {
            start_seconds,
            end_seconds,
        })
    }

    /// Inclusive start of the range, in seconds.
    #[must_use]
    pub const fn start_seconds(self) -> f64 {
        self.start_seconds
    }

    /// Exclusive end of the range, in seconds.
    #[must_use]
    pub const fn end_seconds(self) -> f64 {
        self.end_seconds
    }

    /// Length of the range in seconds.
    #[must_use]
    pub fn duration_seconds(self) -> f64 {
        self.end_seconds - self.start_seconds
    }
}

/// A pair of timestamps that cannot represent a video segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidTimestampRange {
    start_seconds: f64,
    end_seconds: f64,
}

impl InvalidTimestampRange {
    /// Returns the rejected inclusive start.
    #[must_use]
    pub const fn start_seconds(self) -> f64 {
        self.start_seconds
    }

    /// Returns the rejected exclusive end.
    #[must_use]
    pub const fn end_seconds(self) -> f64 {
        self.end_seconds
    }
}

impl fmt::Display for InvalidTimestampRange {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "timestamp range must be finite, non-negative, and non-empty, found {}..{}",
            self.start_seconds, self.end_seconds
        )
    }
}

impl std::error::Error for InvalidTimestampRange {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_accepts_only_finite_positive_values() {
        assert_eq!(
            FramesPerSecond::new(30.0).map(FramesPerSecond::get),
            Ok(30.0)
        );
        for invalid_value in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert!(FramesPerSecond::new(invalid_value).is_err());
        }
    }

    #[test]
    fn timestamp_range_accepts_only_finite_non_empty_ranges() {
        let valid_range = TimestampRange::new(1.25, 2.5).expect("valid timestamp range");
        assert!((valid_range.start_seconds() - 1.25).abs() < f64::EPSILON);
        assert!((valid_range.end_seconds() - 2.5).abs() < f64::EPSILON);
        assert!((valid_range.duration_seconds() - 1.25).abs() < f64::EPSILON);

        for (invalid_start, invalid_end) in [
            (-1.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
        ] {
            assert!(TimestampRange::new(invalid_start, invalid_end).is_err());
        }
    }
}
