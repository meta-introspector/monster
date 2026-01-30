% Git threat model - Prolog analysis
:- module(git_threat_analysis, [
    analyze_dependency/2,
    transitive_threat/2,
    containment_required/2
]).

% Threat indicators from git metadata
threat_indicator(few_commits, 0.3) :- commit_count(Count), Count < 10.
threat_indicator(few_authors, 0.2) :- author_count(Count), Count < 3.
threat_indicator(recent_creation, 0.2) :- repo_age_days(Days), Days < 90.
threat_indicator(no_releases, 0.1) :- release_count(0).

% Threat indicators from source analysis
threat_indicator(unsafe_code, 0.3) :- unsafe_block_count(Count), Count > 5.
threat_indicator(network_heavy, 0.2) :- network_call_count(Count), Count > 10.
threat_indicator(process_spawn, 0.2) :- process_spawn_count(Count), Count > 5.
threat_indicator(file_operations, 0.1) :- file_op_count(Count), Count > 20.
threat_indicator(dangerous_functions, 0.4) :- dangerous_fn_count(Count), Count > 0.

% Aggregate threat score
aggregate_threat_score(Dep, Score) :-
    findall(S, (threat_indicator(_, S), applies_to(Dep)), Scores),
    sum_list(Scores, Score).

% Transitive closure
depends_on_transitive(A, B) :- depends_on(A, B).
depends_on_transitive(A, C) :- depends_on(A, B), depends_on_transitive(B, C).

% Transitive threat
transitive_threat(Dep, TotalScore) :-
    findall(Score, 
        (depends_on_transitive(Dep, SubDep), aggregate_threat_score(SubDep, Score)),
        Scores),
    sum_list(Scores, Sum),
    length(Scores, Count),
    (Count > 0 -> TotalScore is Sum / Count ; TotalScore = 0).

% Containment decision
containment_required(Dep, Zone) :-
    transitive_threat(Dep, Score),
    (Score > 0.8 -> Zone = 71 ;  % Catastrophic
     Score > 0.6 -> Zone = 59 ;  % Critical
     Score > 0.4 -> Zone = 47 ;  % High
     Score > 0.2 -> Zone = 31 ;  % Medium
     Zone = 11).                  % Low

% Analysis report
analyze_dependency(Dep, Report) :-
    aggregate_threat_score(Dep, DirectScore),
    transitive_threat(Dep, TransitiveScore),
    containment_required(Dep, Zone),
    Report = [
        direct_score(DirectScore),
        transitive_score(TransitiveScore),
        containment_zone(Zone)
    ].
