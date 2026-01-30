# Value Lattice Report

Total constants: 19
Unique values: 10

## Most Common Values

- `24` (5x): ["BOSONIC_DIM", "NUM_WORKERS", "NUM_WORKERS", "BOSONIC_DIM", "BOSONIC_DIM"]
- `71` (4x): ["NUM_LAYERS", "PRIME_71", "NUM_SHARDS", "NUM_SHARDS"]
- `440.0` (2x): ["BASE_FREQ", "BASE_FREQ"]
- `70` (2x): ["BASES", "RINGS"]
- `STEPS * BASES * RINGS` (1x): ["TOTAL_ENTRIES"]
- `10.0` (1x): ["DURATION"]
- `808017424794512875886459904961710757005754368000000000` (1x): ["MONSTER_ORDER"]
- `10` (1x): ["STEPS"]
- `44100` (1x): ["SAMPLE_RATE"]
- `10000` (1x): ["BATCH_SIZE"]

## All Constants

- `MONSTER_ORDER` = `808017424794512875886459904961710757005754368000000000` in `src/classification.rs`
- `BOSONIC_DIM` = `24` in `src/bin/self_referential_conformal.rs`
- `NUM_LAYERS` = `71` in `src/bin/bulk_all_layers.rs`
- `NUM_WORKERS` = `24` in `src/bin/vectorize_all_parquets.rs`
- `BATCH_SIZE` = `10000` in `src/bin/vectorize_all_parquets.rs`
- `BASE_FREQ` = `440.0` in `src/bin/monster_harmonics.rs`
- `SAMPLE_RATE` = `44100` in `src/bin/monster_harmonics.rs`
- `DURATION` = `10.0` in `src/bin/monster_harmonics.rs`
- `PRIME_71` = `71` in `src/bin/graded_ring_71.rs`
- `BASE_FREQ` = `440.0` in `src/bin/autolabel_filenames.rs`
- `STEPS` = `10` in `src/bin/monster_walk_gpu.rs`
- `BASES` = `70` in `src/bin/monster_walk_gpu.rs`
- `RINGS` = `70` in `src/bin/monster_walk_gpu.rs`
- `TOTAL_ENTRIES` = `STEPS * BASES * RINGS` in `src/bin/monster_walk_gpu.rs`
- `NUM_WORKERS` = `24` in `src/bin/markov_parquet_shards.rs`
- `BOSONIC_DIM` = `24` in `src/bin/harmonic_folding.rs`
- `NUM_SHARDS` = `71` in `src/bin/harmonic_folding.rs`
- `BOSONIC_DIM` = `24` in `src/bin/lmfdb_inventory_consumer.rs`
- `NUM_SHARDS` = `71` in `src/bin/lmfdb_inventory_consumer.rs`
