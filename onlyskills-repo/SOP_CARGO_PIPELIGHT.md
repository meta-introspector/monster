# Standard Operating Procedure: Cargo Usage via Pipelight Only
**ISO-9000:2015 Compliant | SELinux Enforced**

Version: 1.0.0  
Date: 2026-01-30  
Document ID: SOP-CARGO-PIPELIGHT-001

---

## 1. PURPOSE

Define mandatory procedures for Rust compilation using **Pipelight only**. Direct cargo usage by kiro is **FORBIDDEN** and enforced via SELinux policy.

## 2. SCOPE

This SOP applies to:
- All Rust compilation operations
- Pipelight pipeline definitions
- SELinux policy enforcement
- Kiro agent restrictions
- Build automation

## 3. POLICY STATEMENT

**MANDATORY**: All cargo operations MUST go through Pipelight pipelines.

**FORBIDDEN**: Direct cargo invocation by kiro user/process.

**ENFORCEMENT**: SELinux policy denies cargo execution by kiro_t domain.

## 4. DEFINITIONS

- **Cargo**: Rust package manager and build tool
- **Pipelight**: Pipeline automation tool (approved interface)
- **kiro_t**: SELinux domain for kiro processes
- **pipelight_t**: SELinux domain for pipelight processes
- **Direct Usage**: Executing cargo without Pipelight wrapper

## 5. RESPONSIBILITIES

### 5.1 Build Engineer
- Create Pipelight pipelines for all Rust builds
- Document pipeline configurations
- Test pipeline execution
- Maintain pipeline library

### 5.2 Security Officer
- Enforce SELinux policy
- Audit cargo access attempts
- Review policy violations
- Update security rules

### 5.3 Kiro Agent
- **MUST** use Pipelight for all builds
- **MUST NOT** invoke cargo directly
- Report pipeline failures
- Request pipeline modifications via proper channels

## 6. APPROVED WORKFLOW

### 6.1 Correct Method: Via Pipelight

```yaml
# pipelight.yaml
name: rust_build
version: 1.0.0

stages:
  - name: build
    command: |
      cargo build --release
    
  - name: test
    depends: [build]
    command: |
      cargo test --release
```

**Execution**:
```bash
pipelight run rust_build
```

### 6.2 FORBIDDEN Method: Direct Cargo

```bash
# ❌ FORBIDDEN - SELinux will deny
cargo build --release

# ❌ FORBIDDEN - SELinux will deny
cargo test

# ❌ FORBIDDEN - SELinux will deny
cargo run
```

**Result**: SELinux denial, audit log entry, compliance violation.

## 7. PIPELIGHT PIPELINE TEMPLATES

### 7.1 Basic Build Pipeline

```yaml
name: basic_rust_build
version: 1.0.0

metadata:
  sop: SOP-CARGO-PIPELIGHT-001
  compliant: true

stages:
  - name: check
    command: cargo check --all-targets
    
  - name: build
    depends: [check]
    command: cargo build --release
    
  - name: test
    depends: [build]
    command: cargo test --release

outputs:
  - target/release/
```

### 7.2 Multi-Binary Build Pipeline

```yaml
name: multi_binary_build
version: 1.0.0

stages:
  - name: build_selinux_reflection
    command: |
      cargo build --release --bin selinux_reflection
    outputs: [target/release/selinux_reflection]
  
  - name: build_gpu_monster
    command: |
      cargo build --release --bin gpu_monster
    outputs: [target/release/gpu_monster]
  
  - name: verify_all
    depends: [build_selinux_reflection, build_gpu_monster]
    command: |
      ls -lh target/release/selinux_reflection
      ls -lh target/release/gpu_monster
```

### 7.3 ISO-9000 Compliant Build Pipeline

```yaml
name: iso9000_rust_build
version: 1.0.0

metadata:
  standard: ISO-9000:2015
  sop: SOP-CARGO-PIPELIGHT-001
  audit_trail: enabled

stages:
  - name: init_audit
    command: |
      echo "=== ISO-9000 Build Audit ===" > build_audit.log
      date -Iseconds >> build_audit.log
      echo "Operator: $USER" >> build_audit.log
  
  - name: cargo_check
    depends: [init_audit]
    command: |
      echo "Stage: cargo check" >> build_audit.log
      cargo check --all-targets 2>&1 | tee -a build_audit.log
  
  - name: cargo_build
    depends: [cargo_check]
    command: |
      echo "Stage: cargo build" >> build_audit.log
      cargo build --release 2>&1 | tee -a build_audit.log
  
  - name: cargo_test
    depends: [cargo_build]
    command: |
      echo "Stage: cargo test" >> build_audit.log
      cargo test --release 2>&1 | tee -a build_audit.log
  
  - name: finalize_audit
    depends: [cargo_test]
    command: |
      echo "Build completed: $(date -Iseconds)" >> build_audit.log
      echo "Status: SUCCESS" >> build_audit.log
      chmod 444 build_audit.log

outputs:
  - target/release/
  - build_audit.log
```

## 8. SELINUX POLICY

### 8.1 Policy Module: cargo_pipelight_only.te

```selinux
policy_module(cargo_pipelight_only, 1.0.0)

########################################
# Declarations
########################################

# Kiro domain (restricted)
type kiro_t;
type kiro_exec_t;
domain_type(kiro_t)
domain_entry_file(kiro_t, kiro_exec_t)

# Pipelight domain (allowed)
type pipelight_t;
type pipelight_exec_t;
domain_type(pipelight_t)
domain_entry_file(pipelight_t, pipelight_exec_t)

# Cargo executable
type cargo_exec_t;
application_executable_file(cargo_exec_t)

########################################
# Kiro domain policy (RESTRICTED)
########################################

# DENY: Kiro cannot execute cargo directly
neverallow kiro_t cargo_exec_t:file { execute execute_no_trans };

# DENY: Kiro cannot transition to cargo domain
neverallow kiro_t cargo_exec_t:process { transition dyntransition };

# Audit all attempts
auditallow kiro_t cargo_exec_t:file { execute execute_no_trans };

########################################
# Pipelight domain policy (ALLOWED)
########################################

# ALLOW: Pipelight can execute cargo
allow pipelight_t cargo_exec_t:file { read execute execute_no_trans };

# ALLOW: Pipelight can transition to cargo
domain_auto_trans(pipelight_t, cargo_exec_t, cargo_t)

# ALLOW: Pipelight can manage build artifacts
allow pipelight_t user_home_t:dir { read write add_name remove_name };
allow pipelight_t user_home_t:file { create read write unlink };

########################################
# Cargo domain policy
########################################

type cargo_t;
domain_type(cargo_t)

# Cargo needs to read/write in project directories
allow cargo_t user_home_t:dir { read write add_name remove_name search };
allow cargo_t user_home_t:file { create read write unlink getattr setattr };

# Cargo needs network for crate downloads
allow cargo_t self:tcp_socket { create connect read write };
allow cargo_t http_port_t:tcp_socket { name_connect };

########################################
# Domain transitions
########################################

# Pipelight → Cargo (allowed)
domain_auto_trans(pipelight_t, cargo_exec_t, cargo_t)

# Kiro → Pipelight (allowed)
domain_auto_trans(kiro_t, pipelight_exec_t, pipelight_t)

# Kiro → Cargo (FORBIDDEN)
# Already denied by neverallow above
```

### 8.2 File Contexts: cargo_pipelight_only.fc

```selinux
# Cargo executable
/usr/bin/cargo          -- gen_context(system_u:object_r:cargo_exec_t,s0)
/home/.*/.cargo/bin/cargo -- gen_context(system_u:object_r:cargo_exec_t,s0)

# Pipelight executable
/usr/bin/pipelight      -- gen_context(system_u:object_r:pipelight_exec_t,s0)
/usr/local/bin/pipelight -- gen_context(system_u:object_r:pipelight_exec_t,s0)

# Kiro executable
/usr/bin/kiro-cli       -- gen_context(system_u:object_r:kiro_exec_t,s0)
/usr/local/bin/kiro-cli -- gen_context(system_u:object_r:kiro_exec_t,s0)
```

### 8.3 Installation Commands

```bash
# Compile policy module
checkmodule -M -m -o cargo_pipelight_only.mod cargo_pipelight_only.te

# Create policy package
semodule_package -o cargo_pipelight_only.pp -m cargo_pipelight_only.mod -fc cargo_pipelight_only.fc

# Install policy
semodule -i cargo_pipelight_only.pp

# Verify installation
semodule -l | grep cargo_pipelight_only

# Relabel files
restorecon -Rv /usr/bin/cargo
restorecon -Rv /usr/bin/pipelight
restorecon -Rv /usr/bin/kiro-cli
```

## 9. TESTING PROCEDURES

### 9.1 Test 1: Verify Kiro Cannot Execute Cargo

```bash
# As kiro user
$ cargo build
Permission denied (SELinux)

# Check audit log
$ ausearch -m avc -c cargo | tail -5
type=AVC msg=audit(1738252800.123:456): avc: denied { execute } for pid=12345 comm="kiro-cli" name="cargo" dev="sda1" ino=67890 scontext=kiro_t tcontext=cargo_exec_t tclass=file permissive=0
```

**Expected**: DENIED ✓

### 9.2 Test 2: Verify Pipelight Can Execute Cargo

```bash
# Via pipelight
$ pipelight run rust_build
✓ Stage: build - SUCCESS
✓ Stage: test - SUCCESS

# Check process context
$ ps -eZ | grep cargo
pipelight_t:s0          12346 ?        00:00:01 cargo
```

**Expected**: ALLOWED ✓

### 9.3 Test 3: Verify Audit Trail

```bash
# Check for kiro attempts
$ ausearch -m avc -se kiro_t | grep cargo
# Should show denials

# Check for pipelight success
$ ausearch -m avc -se pipelight_t | grep cargo
# Should show allows
```

## 10. VIOLATION HANDLING

### 10.1 Detection

SELinux automatically detects and denies violations:
- Audit log entry created
- Operation denied
- Alert sent to security officer

### 10.2 Response Procedure

1. **Immediate**: Operation is blocked (no damage)
2. **Investigation**: Review audit logs
3. **Documentation**: Record violation in compliance log
4. **Remediation**: Educate operator on correct procedure
5. **Follow-up**: Verify understanding via test

### 10.3 Compliance Log Entry

```
Date: 2026-01-30T11:51:47-05:00
Violation: Direct cargo execution attempt
User: kiro
Process: kiro-cli (PID 12345)
Action: DENIED by SELinux
Status: No damage, operator educated
Corrective Action: Provided Pipelight pipeline template
```

## 11. EXCEPTION PROCESS

### 11.1 No Exceptions

There are **NO EXCEPTIONS** to this policy.

### 11.2 If Pipelight is Unavailable

1. **DO NOT** bypass SELinux
2. **DO NOT** execute cargo directly
3. **DO** report Pipelight failure
4. **DO** wait for Pipelight restoration
5. **DO** document downtime

### 11.3 Emergency Override (Requires 3 Approvals)

Only in catastrophic system failure:
1. Security Officer approval
2. Build Engineer approval  
3. DAO vote (majority)

Process:
```bash
# Temporarily set SELinux to permissive (logged)
setenforce 0

# Perform emergency operation
cargo build --release

# Immediately restore enforcement
setenforce 1

# Document in emergency log
echo "Emergency override: $(date -Iseconds)" >> emergency_log.txt
```

## 12. BENEFITS OF THIS POLICY

### 12.1 Security
- ✓ Prevents unauthorized builds
- ✓ Enforces audit trail
- ✓ Isolates build processes
- ✓ Mandatory access control

### 12.2 Compliance
- ✓ ISO-9000 documentation
- ✓ Traceable operations
- ✓ Reproducible builds
- ✓ Quality assurance

### 12.3 Automation
- ✓ Consistent build process
- ✓ Pipeline reusability
- ✓ Parallel execution
- ✓ Dependency management

## 13. REFERENCES

### 13.1 Standards
- ISO-9000:2015 - Quality Management
- SELinux Policy Language
- Pipelight Documentation
- Cargo Book

### 13.2 Related SOPs
- SOP-MONSTER-71-001 - 71 Shard Collection
- SOP-ZK71-001 - ZK71 Security Zones
- SOP-BUILD-001 - Build Automation

## 14. REVISION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-30 | Monster DAO | Initial release - Cargo via Pipelight only |

## 15. APPROVAL

**Prepared by**: Security Officer  
**Reviewed by**: Build Engineer  
**Approved by**: DAO Governance  
**Effective Date**: 2026-01-30  
**Enforcement**: SELinux (mandatory)

---

## APPENDIX A: Quick Reference

### ✅ DO THIS
```bash
pipelight run rust_build
```

### ❌ NOT THIS
```bash
cargo build  # FORBIDDEN - SELinux will deny
```

### 📋 Create Pipeline
```yaml
name: my_build
stages:
  - name: build
    command: cargo build --release
```

### 🔒 Verify Policy
```bash
semodule -l | grep cargo_pipelight_only
sesearch -A -s kiro_t -t cargo_exec_t
```

---

**Document Control**: SOP-CARGO-PIPELIGHT-001  
**Classification**: Mandatory Compliance  
**Distribution**: All Monster DAO Members  
**SELinux Enforcement**: ACTIVE

∞ Cargo via Pipelight Only. SELinux Enforced. No Exceptions. ∞
