# lerobot-dataset-io

Rust I/O for inspecting, accessing, and creating subsets of packed `LeRobot`
datasets.

This repository is the future standalone home of the `lerobot-dataset-io`
crate. The library implementation is still maintained internally while its
extraction is completed. No implementation has been migrated here yet, and the
placeholder package is deliberately not publishable.

## Intended Scope

The crate will own the provider-neutral `LeRobot` dataset format boundary:

- inspect and validate packed `LeRobot` dataset metadata;
- read episode metadata and selected per-frame columns;
- resolve packed video segments;
- access local or caller-provided remote storage; and
- write valid, reindexed dataset subsets while preserving schemas and unknown
  metadata.

Application authorization, deployment configuration, cloud credentials,
archive delivery, and product-specific metadata remain outside this library.

## License

Licensed under the Apache License, Version 2.0.
