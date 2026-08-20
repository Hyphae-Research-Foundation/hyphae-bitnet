# Upstream

Celiums BitNet is derived from [Microsoft BitNet](https://github.com/microsoft/BitNet).

## Baseline

- Microsoft BitNet: `0b341e582afbf9e1011f24744b554c96a3477eb5`
- llama.cpp dependency: `390c307752ab78fd8189f359d6954c9ba1be74af`
- Archived Celiums engine repository (provenance only):
  `https://github.com/celiumsai/celiums-bitnet-llama`

## Update Policy

Upstream changes are reviewed and imported manually. The engine is vendored
in-tree at `3rdparty/llama.cpp`; the snapshot's engine commit is recorded in
`3rdparty/llama.cpp/ENGINE_COMMIT`, while `cmake/ENGINE_TREE` records the exact
vendored tree. A normal clone, build, test, and package operation must not access
the archived repository.
After reviewing an intentional engine change, update the pin with
`scripts/compute-engine-tree.sh` and record its output in `cmake/ENGINE_TREE`.

An upstream update must pass the strict numerical tests, model perplexity tests,
and hardware benchmark matrix before it can enter a release.

## Divergence

Celiums maintains corrections and optimizations that may not exist in the
Microsoft repository or its llama.cpp dependency. See `CHANGES.md` and
`docs/NUMERICAL_CONTRACT.md`.

## Licensing

Celiums contributions are Apache-2.0. Upstream Microsoft BitNet code remains
under its retained MIT license, and the llama.cpp dependency retains its own
MIT and third-party notices. See `LICENSE`, `LICENSE-MIT`, and `NOTICE`.
