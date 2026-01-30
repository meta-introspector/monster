# SOP: Anti-Python Policy for Kiro-CLI
**Based on: PYTHON_REMOVAL.md**  
**Enforcement: SELinux Mandatory Access Control**

Version: 1.0.0  
Date: 2026-01-30  
Document ID: SOP-NO-PYTHON-001

---

## 1. POLICY STATEMENT

**Python is FORBIDDEN for kiro-cli.**

All functionality MUST be implemented in:
- ✅ Rust
- ✅ Nix
- ✅ Prolog
- ✅ Lean4
- ✅ MiniZinc
- ✅ Circom

**NO EXCEPTIONS.**

## 2. RATIONALE

From PYTHON_REMOVAL.md:
- Python removed from all tracked files (2026-01-29)
- 23 Python files moved to quarantine/
- Pure Rust + Nix workflow established
- Pre-commit hook enforces Rust-only

**Benefits**:
- ✓ Reproducible builds
- ✓ Type safety
- ✓ Performance
- ✓ No dependency hell
- ✓ Memory safety

## 3. SELINUX ENFORCEMENT

### Policy Module: no_python_kiro.te

**NEVERALLOW rules**:
```selinux
neverallow kiro_t python_exec_t:file { execute execute_no_trans };
neverallow kiro_t python_exec_t:process { transition dyntransition };
neverallow kiro_t python_script_t:file { execute execute_no_trans };
```

**Result**: Any Python execution attempt by kiro → DENIED

### Installation

```bash
# Compile policy
checkmodule -M -m -o no_python_kiro.mod no_python_kiro.te

# Create package
semodule_package -o no_python_kiro.pp \
  -m no_python_kiro.mod \
  -fc no_python_kiro.fc

# Install (requires root)
sudo semodule -i no_python_kiro.pp

# Verify
semodule -l | grep no_python_kiro

# Relabel
sudo restorecon -Rv /usr/bin/python*
sudo restorecon -Rv /usr/bin/kiro-cli
```

## 4. WHAT HAPPENS IF KIRO TRIES PYTHON

### Attempt
```bash
$ python3 script.py
```

### Result
```
Permission denied (SELinux)
```

### Audit Log
```
type=AVC msg=audit(...): avc: denied { execute } 
  for pid=... comm="kiro-cli" name="python3" 
  scontext=kiro_t tcontext=python_exec_t 
  tclass=file permissive=0

Message: DENIED: Python forbidden for kiro-cli per PYTHON_REMOVAL.md
         Use Rust + Nix + Prolog + Lean4 instead
         See: SOP-CARGO-PIPELIGHT-001, PYTHON_REMOVAL.md
```

## 5. APPROVED ALTERNATIVES

### Instead of Python, use:

**Data Processing**: Rust + Polars
```rust
use polars::prelude::*;
let df = CsvReader::from_path("data.csv")?.finish()?;
```

**Scripting**: Nix
```nix
{ pkgs ? import <nixpkgs> {} }:
pkgs.runCommand "process" {} ''
  # Pure Nix script
''
```

**Logic**: Prolog
```prolog
process_data(Input, Output) :-
    % Pure logic
    transform(Input, Output).
```

**Proofs**: Lean4
```lean
theorem data_valid : ∀ x, process x = valid := by
  intro x
  simp [process]
```

## 6. QUARANTINE ZONE

Python files preserved in `quarantine/scripts/`:
- ✓ Not tracked by git
- ✓ Read-only access for kiro
- ✓ Cannot execute
- ✓ Reference only

**SELinux label**: `python_quarantine_t`

## 7. COMPLIANCE VERIFICATION

### Check 1: No Python in tracked files
```bash
git ls-files | grep '\.py$'
# Expected: (empty)
```

### Check 2: SELinux policy active
```bash
semodule -l | grep no_python_kiro
# Expected: no_python_kiro 1.0.0
```

### Check 3: Kiro cannot execute Python
```bash
sudo -u kiro python3 --version
# Expected: Permission denied
```

### Check 4: Audit log shows denials
```bash
ausearch -m avc -c kiro-cli | grep python
# Expected: denied { execute }
```

## 8. MIGRATION GUIDE

### Old (Python)
```python
def process_data(input_file):
    with open(input_file) as f:
        data = json.load(f)
    return analyze(data)
```

### New (Rust)
```rust
fn process_data(input_file: &str) -> Result<Analysis> {
    let data: Data = serde_json::from_reader(
        File::open(input_file)?
    )?;
    Ok(analyze(&data))
}
```

## 9. ENFORCEMENT LEVELS

### Level 1: Pre-commit Hook
- Blocks Python commits
- Enforces Rust-only

### Level 2: SELinux
- Blocks Python execution
- Mandatory access control

### Level 3: Code Review
- Rejects Python PRs
- Requires Rust alternatives

### Level 4: CI/CD
- Fails on Python detection
- Rust-only builds

## 10. EXCEPTIONS

**There are NO exceptions.**

If you need Python functionality:
1. Implement in Rust
2. Use Nix for scripting
3. Use Prolog for logic
4. Use Lean4 for proofs

## 11. REFERENCES

- PYTHON_REMOVAL.md - Original removal documentation
- SOP-CARGO-PIPELIGHT-001 - Cargo via Pipelight only
- PURE_RUST_DEPLOYMENT.md - Pure Rust deployment guide
- .git/hooks/pre-commit - Rust-only pre-commit hook

---

**Document Control**: SOP-NO-PYTHON-001  
**Classification**: Mandatory Compliance  
**Distribution**: All Kiro Users  
**SELinux Enforcement**: ACTIVE

∞ No Python. Rust Only. SELinux Enforced. No Exceptions. ∞
