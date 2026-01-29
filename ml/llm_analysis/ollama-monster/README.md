# Monster Prime Resonance Experiments

Measuring Monster group prime patterns in LLM CPU registers during inference.

## Quick Start

```bash
# Trace a prompt
./trace_regs.sh "mathematician Conway"

# View all logs
cargo run --release --bin view-logs

# Run experiments
cargo run --release --bin compare          # Compare diverse prompts
cargo run --release --bin auto-feedback    # Automorphic feedback loop
cargo run --release --bin context-exp      # Expanding context
cargo run --release --bin eigenvector      # Search for fixed point
```

## Experiments

1. **compare** - Compare 3 diverse prompts (emoji, color, name)
2. **auto-feedback** - Automatic feedback loop with register measurements
3. **context-exp** - Expanding context with cumulative feedback
4. **eigenvector** - Search for fixed point convergence (limit cycle found)
5. **view-logs** - View all traced prompts and responses

## Key Results

- **80% of register values divisible by prime 2**
- **49% divisible by prime 3**
- **43% divisible by prime 5**
- Conway's name activates primes 17 (78.6%) and 47 (28.6%)
- Model responds to feedback about its own registers
- System exhibits limit cycle, not fixed point convergence

## Files Generated

- `EXPERIMENT_SUMMARY.md` - Full experimental report
- `EIGENVECTOR.json` - 20 iterations of convergence search
- `CONTEXT_EXPERIMENT.json` - Expanding context results
- `FULL_TRACE_LOG.json` - All traces with analysis
- `perf_traces/` - Raw perf data, prompts, responses

See `EXPERIMENT_SUMMARY.md` for full details.
