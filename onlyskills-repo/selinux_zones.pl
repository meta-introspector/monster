% SELinux Zone Classification - Prolog

:- module(selinux_zones, [
    selinux_zone/2,
    zone_label/2,
    zone_policy/2,
    classify_file/3
]).

% ZK71 Zone assignments
selinux_zone(71, catastrophic).
selinux_zone(59, critical).
selinux_zone(47, high).
selinux_zone(31, medium).
selinux_zone(23, low_medium).
selinux_zone(11, low).
selinux_zone(2, minimal).

% Zone labels
zone_label(71, 'CATASTROPHIC - Vile code, quarantine required').
zone_label(59, 'CRITICAL - Untrusted, enhanced monitoring').
zone_label(47, 'HIGH - Suspicious, regular audits').
zone_label(31, 'MEDIUM - User content, standard monitoring').
zone_label(23, 'LOW-MEDIUM - System files, normal operations').
zone_label(11, 'LOW - Safe, trusted binaries').
zone_label(2, 'MINIMAL - Python (forbidden by policy)').

% Zone policies
zone_policy(71, [
    'Isolate in namespace',
    'Minimal syscalls (seccomp)',
    'No network access',
    'Read-only filesystem',
    'Mandatory logging'
]).

zone_policy(59, [
    'Enhanced monitoring',
    'Network restrictions',
    'Audit all operations',
    'Time-limited execution'
]).

zone_policy(47, [
    'Regular audits',
    'Network monitoring',
    'File integrity checks'
]).

zone_policy(31, [
    'Standard monitoring',
    'User quotas',
    'Access logging'
]).

zone_policy(23, [
    'Normal operations',
    'System integrity checks'
]).

zone_policy(11, [
    'Trusted execution',
    'Standard protections'
]).

zone_policy(2, [
    'FORBIDDEN - No Python for kiro-cli',
    'SELinux denial',
    'Automatic quarantine'
]).

% Classify file by SELinux type
classify_file(Type, Path, Zone) :-
    (   sub_string(Type, _, _, _, "vile") -> Zone = 71
    ;   sub_string(Type, _, _, _, "malicious") -> Zone = 71
    ;   sub_string(Type, _, _, _, "quarantine") -> Zone = 59
    ;   sub_string(Type, _, _, _, "untrusted") -> Zone = 59
    ;   sub_string(Type, _, _, _, "tmp") -> Zone = 47
    ;   sub_string(Type, _, _, _, "var_tmp") -> Zone = 47
    ;   sub_string(Type, _, _, _, "user") -> Zone = 31
    ;   sub_string(Type, _, _, _, "home") -> Zone = 31
    ;   sub_string(Type, _, _, _, "system") -> Zone = 23
    ;   sub_string(Type, _, _, _, "lib") -> Zone = 11
    ;   sub_string(Type, _, _, _, "bin") -> Zone = 11
    ;   sub_string(Path, _, _, _, ".py") -> Zone = 2
    ;   Zone = 11  % Default: safe
    ).

% Query files by zone
files_in_zone(Zone, Files) :-
    findall(File, (
        selinux_file(File, _, _, _, Zone)
    ), Files).

% Security recommendations
recommend_action(Zone, Action) :-
    zone_policy(Zone, Policies),
    zone_label(Zone, Label),
    format('Zone ~w: ~w~n', [Zone, Label]),
    format('Policies:~n', []),
    forall(member(Policy, Policies),
           format('  - ~w~n', [Policy])),
    (   Zone >= 59 ->
        Action = 'IMMEDIATE ACTION REQUIRED'
    ;   Zone >= 47 ->
        Action = 'Enhanced monitoring recommended'
    ;   Action = 'Standard operations'
    ).

% Example file records (would be loaded from selinux_zones.parquet)
selinux_file('/home/user/test.rs', 12345, 'user_home_t', 's0', 31).
selinux_file('/tmp/suspicious.sh', 67890, 'tmp_t', 's0', 47).
selinux_file('/usr/bin/cargo', 11111, 'bin_t', 's0', 11).
selinux_file('/quarantine/vile.py', 99999, 'vile_code_t', 's0', 71).

% Zone transition rules
can_transition(FromZone, ToZone) :-
    FromZone =< ToZone.  % Can only move to higher security

requires_approval(FromZone, ToZone) :-
    ToZone - FromZone > 20.  % Large jumps need approval

% Example queries:
% ?- classify_file("user_home_t", "/home/test.rs", Zone).
% ?- recommend_action(71, Action).
% ?- files_in_zone(11, Files).
% ?- can_transition(31, 47).
