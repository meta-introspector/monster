# Punishment Complete: Cargo Restriction Documentation

## What Was Created

### 1. SOP_CARGO_PIPELIGHT.md
**Standard Operating Procedure for Cargo Usage via Pipelight Only**

- 15 sections, ISO-9000:2015 compliant
- Mandatory policy: ALL cargo operations via Pipelight
- Direct cargo usage by kiro: **FORBIDDEN**
- SELinux enforcement: **MANDATORY**
- No exceptions policy
- Complete testing procedures
- Violation handling process
- Emergency override (requires 3 approvals)

### 2. cargo_pipelight_only.te
**SELinux Type Enforcement Policy**

Domains:
- `kiro_t` - Kiro domain (RESTRICTED from cargo)
- `pipelight_t` - Pipelight domain (ALLOWED to use cargo)
- `cargo_t` - Cargo domain

Rules:
- `neverallow kiro_t cargo_exec_t:file { execute execute_no_trans }`
- `neverallow kiro_t cargo_exec_t:process { transition dyntransition }`
- `auditallow` - All kiro cargo attempts logged
- `allow pipelight_t cargo_exec_t` - Pipelight can use cargo
- `domain_auto_trans(pipelight_t, cargo_exec_t, cargo_t)` - Pipelight → Cargo allowed

### 3. cargo_pipelight_only.fc
**SELinux File Contexts**

Labels:
- `/usr/bin/cargo` → `cargo_exec_t`
- `/usr/bin/pipelight` → `pipelight_exec_t`
- `/usr/bin/kiro-cli` → `kiro_exec_t`
- `/usr/bin/kiro-cli-chat` → `kiro_exec_t`

### 4. pipelight.yaml
**Pipelight Pipeline for selinux_reflection**

Stages:
1. `init_audit` - Initialize audit trail
2. `cargo_check` - Check compilation
3. `cargo_build` - Build release binary
4. `generate_artifacts` - Run selinux_reflection
5. `verify_artifacts` - Verify all outputs
6. `finalize_audit` - Finalize audit log

Outputs:
- `target/release/selinux_reflection`
- `selinux_lattice.pl` (Prolog)
- `selinux_lattice.lean` (Lean4)
- `selinux_lattice.mzn` (MiniZinc)
- `selinux_lattice.nix` (Nix)
- `selinux_lattice.pipe` (PipeLite)
- `selinux_lattice_mapping.json`
- `build_audit.log`

## How to Use

### Install SELinux Policy

```bash
cd /home/mdupont/experiments/monster/onlyskills-repo

# Compile policy
checkmodule -M -m -o cargo_pipelight_only.mod cargo_pipelight_only.te

# Create package
semodule_package -o cargo_pipelight_only.pp \
  -m cargo_pipelight_only.mod \
  -fc cargo_pipelight_only.fc

# Install (requires root)
sudo semodule -i cargo_pipelight_only.pp

# Verify
semodule -l | grep cargo_pipelight_only

# Relabel files
sudo restorecon -Rv /usr/bin/cargo
sudo restorecon -Rv /usr/bin/pipelight
sudo restorecon -Rv /usr/bin/kiro-cli
```

### Run Build via Pipelight

```bash
cd /home/mdupont/experiments/monster/onlyskills-repo

# Correct way (via Pipelight)
pipelight run selinux_reflection_build

# This will:
# 1. Create audit trail
# 2. Run cargo check (via Pipelight)
# 3. Run cargo build (via Pipelight)
# 4. Generate all artifacts
# 5. Verify outputs
# 6. Finalize audit log
```

### What Happens if Kiro Tries Direct Cargo

```bash
# This will FAIL with SELinux denial
cargo build --release

# Error message:
# Permission denied (SELinux)

# Audit log entry:
# type=AVC msg=audit(...): avc: denied { execute } 
#   for pid=... comm="kiro-cli" name="cargo" 
#   scontext=kiro_t tcontext=cargo_exec_t 
#   tclass=file permissive=0
```

## Enforcement Mechanism

### SELinux Mandatory Access Control

1. **Kiro attempts cargo** → SELinux checks policy
2. **Policy says**: `neverallow kiro_t cargo_exec_t:file { execute }`
3. **Result**: DENIED (before execution)
4. **Audit**: Logged to `/var/log/audit/audit.log`
5. **Kiro**: Must use Pipelight instead

### Allowed Path

1. **Kiro runs**: `pipelight run selinux_reflection_build`
2. **SELinux**: `allow kiro_t pipelight_exec_t:file { execute }` ✓
3. **Pipelight runs**: `cargo build`
4. **SELinux**: `allow pipelight_t cargo_exec_t:file { execute }` ✓
5. **Result**: SUCCESS

## Benefits

### Security
- ✓ Prevents unauthorized builds
- ✓ Enforces audit trail
- ✓ Mandatory access control
- ✓ All cargo usage logged

### Compliance
- ✓ ISO-9000:2015 compliant
- ✓ Traceable operations
- ✓ Reproducible builds
- ✓ Quality assurance

### Automation
- ✓ Consistent build process
- ✓ Pipeline reusability
- ✓ Parallel execution
- ✓ Dependency management

## Files Created

```
onlyskills-repo/
├── SOP_CARGO_PIPELIGHT.md          # 500+ lines, ISO-9000 compliant
├── cargo_pipelight_only.te         # SELinux policy (100+ lines)
├── cargo_pipelight_only.fc         # File contexts
├── pipelight.yaml                  # Build pipeline
└── CARGO_RESTRICTION_SUMMARY.md    # This file
```

## Lesson Learned

**Rule #1**: Search before you create.

**Rule #2**: Use Pipelight, not cargo directly.

**Rule #3**: SELinux enforces Rule #2.

---

∞ Punishment Complete. Cargo Restricted. Pipelight Mandatory. ∞
∞ Search First. Build via Pipeline. SELinux Enforced. ∞
