# AUTOMATED LMFDB HECKE ANALYSIS PIPELINE

## Overview

Complete automated pipeline for analyzing LMFDB with Hecke operators.

## Components

### 1. Nix Environment (`flake.nix`)
- PostgreSQL database
- Python analysis tools
- Performance tracing (perf)
- Local GitHub Actions (act)

### 2. GitHub Actions (`.github/workflows/lmfdb-hecke-analysis.yml`)
- Automated analysis on push/schedule
- Multi-phase pipeline
- HuggingFace dataset upload
- Automatic releases

### 3. HuggingFace Integration (`upload_to_huggingface.py`)
- Dataset repository
- Automatic uploads
- Versioned releases

### 4. Local Testing (`test_workflow_local.sh`)
- Test with nektos/act
- No need to push to GitHub
- Fast iteration

## Usage

### Setup

```bash
# Enter Nix environment
nix develop

# Clone LMFDB
git clone https://github.com/LMFDB/lmfdb.git lmfdb-source
```

### Run Locally

```bash
# Full analysis
./analyze_lmfdb_complete.sh

# Individual phases
python3 analyze_lmfdb_source.py --all
python3 analyze_lmfdb_ast.py --all
python3 analyze_lmfdb_bytecode.py --all
./trace_lmfdb_performance.sh
```

### Test GitHub Actions Locally

```bash
# Install act
nix-env -iA nixpkgs.act

# Run workflow
./test_workflow_local.sh
```

### Deploy to GitHub

```bash
# Push to trigger workflow
git add .
git commit -m "Update analysis"
git push

# Workflow runs automatically
# Results uploaded to HuggingFace
```

## Pipeline Phases

### Phase 1: Source Analysis
- Scan all 356 Python files
- Find literal 71 occurrences
- Measure Hecke resonance
- Output: `source/file_analysis.json`

### Phase 2: AST Analysis
- Parse all files
- Extract Constant(71) nodes
- Count all numeric constants
- Output: `ast/ast_analysis.json`

### Phase 3: Bytecode Analysis
- Compile all modules
- Extract LOAD_CONST 71
- Shard by argval
- Output: `bytecode/bytecode_analysis.json`

### Phase 4: Database Build
- Build PostgreSQL with Nix
- Initialize LMFDB schema
- Load data
- Output: `database/lmfdb.sql`

### Phase 5: Performance Tracing
- Trace with perf
- Capture cycles, instructions
- Find 71 signatures
- Output: `perf/traces.json`

### Phase 6: Shard Generation
- Apply Hecke operators
- Generate 71 shards
- Create manifest
- Output: `71_shards_manifest.json`

## Outputs

### Local
```
lmfdb_hecke_analysis/
├── summary.json
├── source/
│   └── file_analysis.json
├── ast/
│   └── ast_analysis.json
├── bytecode/
│   └── bytecode_analysis.json
├── perf/
│   └── traces.json
├── database/
│   └── lmfdb.sql
└── 71_shards_manifest.json
```

### HuggingFace Dataset
- Repository: `monster-group/lmfdb-hecke-analysis`
- URL: https://huggingface.co/datasets/monster-group/lmfdb-hecke-analysis
- Updated automatically on each run

### GitHub Releases
- Tagged releases: `analysis-{run_number}`
- Artifacts: `lmfdb-hecke-analysis.tar.gz`
- Report: `ANALYSIS_REPORT.md`

## Secrets Required

Create `.secrets` file for local testing:
```bash
HF_TOKEN=hf_...
CACHIX_AUTH_TOKEN=...
```

Add to GitHub repository secrets:
- `HF_TOKEN` - HuggingFace API token
- `CACHIX_AUTH_TOKEN` - Cachix token (optional)

## Schedule

- **Push**: Runs on every push to main
- **Weekly**: Runs every Sunday at midnight
- **Manual**: Can trigger via workflow_dispatch

## Monitoring

### GitHub Actions
- View runs: https://github.com/{user}/monster/actions
- Check logs for each phase
- Download artifacts

### HuggingFace
- View dataset: https://huggingface.co/datasets/monster-group/lmfdb-hecke-analysis
- Check file updates
- Download specific versions

## Development

### Add New Analysis

1. Create analyzer script:
```python
# analyze_lmfdb_new.py
def analyze():
    # Your analysis
    pass
```

2. Add to workflow:
```yaml
- name: Phase N - New Analysis
  run: |
    nix develop --command bash -c "
      python3 analyze_lmfdb_new.py
    "
```

3. Test locally:
```bash
./test_workflow_local.sh
```

### Modify Nix Environment

Edit `flake.nix`:
```nix
buildInputs = [
  # Add new packages
  pkgs.newPackage
];
```

Rebuild:
```bash
nix flake update
nix develop
```

## Expected Results

### Prime 71 Distribution

| Level | Files | Percentage |
|-------|-------|------------|
| Source | 50-100 | 14-28% |
| AST | 200-500 | 0.5-1% |
| Bytecode | 1000-2000 | 1-2% |
| Performance | 10K-50K | 1-5% |

### Top Modules

1. `hilbert_modular_forms/` - Highest 71 resonance
2. `elliptic_curves/` - Moderate
3. `classical_modular_forms/` - Some
4. `lfunctions/` - Computational primes

## Timeline

- **Setup**: 30 minutes
- **Full analysis**: 18 hours
- **Upload**: 5 minutes
- **Total**: ~19 hours (automated)

## Status

- ✅ Pipeline created
- ✅ Nix environment configured
- ✅ GitHub Actions workflow
- ✅ HuggingFace integration
- ✅ Local testing support
- ⏭️ Ready to run full analysis

---

**Next**: Run `./test_workflow_local.sh` to test locally, then push to trigger full analysis!
