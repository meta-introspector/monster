# Datasets

Organized data files from Monster Group analysis.

## Structure

- `traces/` - C source files for execution tracing
- `execution_traces/` - Parquet files with execution data
- `harmonics/` - Harmonic analysis scan results
- `benchmarks/` - Performance benchmarks
- `analysis/` - JSON analysis results

## Files

### Execution Traces
- `trace_71_from_71.parquet` - Trace starting from prime 71
- `trace_71_from_multi.parquet` - Trace from multi-prime value

### Harmonics
- `harmonics_scan.parquet` - Scan of 320 Julia files
- `harmonics_ranked.parquet` - Files ranked by relevance
- `harmonics_scan_results.json` - Detailed scan results

### Analysis
- `operations_around_71.json` - Perf trace analysis
- `operations_on_71_analysis.json` - Complete operations analysis

### Benchmarks
- `monster_code_analysis.parquet` - Code metrics
