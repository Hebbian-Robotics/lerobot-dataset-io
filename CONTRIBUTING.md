# Contributing

## Code Quality

```bash
lychee -v .     # Lint Markdown files for broken links
```

### Rust

```bash
cargo fmt
cargo clippy --all-targets --all-features
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

## Code Style & Philosophy

### Typing & Pattern Matching

- Prefer explicit types over unstructured values; make invalid states
  unrepresentable where practical.
- Prefer typed variants over string literals when the set of valid values is
  known.
- Use exhaustive pattern matching so the compiler verifies that every variant
  is handled.
- Keep each public export explicitly defined for clear IDE navigation and
  readable documentation.

### Forward Compatibility

- Preserve unknown format fields and variants when the underlying LeRobot
  format permits round trips.
- Keep application policy, credentials, and deployment configuration outside
  the format library.

### Tests

- Test observable format and business outcomes, not implementation details.
- Do not add tests whose primary assertion is that a mock or a third-party
  library works.
- Prefer small generated fixtures that demonstrate a format invariant over
  copied production datasets.

### Self-Documenting Code

- Use descriptive function and variable names that read like documentation.
- Comment non-obvious invariants and architectural decisions; do not restate
  what the code already shows.
