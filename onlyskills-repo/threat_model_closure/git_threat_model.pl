% Git dependency threat model
:- module(git_threat_model, [
    dependency/5,
    threat_score/2,
    transitive_threat/2
]).

% dependency(Name, Version, RepoURL, CommitCount, AuthorCount)
dependency('serde', '*', 'https://github.com/rust-lang/serde', 0, 0).
dependency('serde_json', '*', 'https://github.com/rust-lang/serde_json', 0, 0).

% Threat scores
threat_score('serde', 0.5).
threat_score('serde_json', 0.5).

% High threat if score > 0.7
high_threat(Dep) :- threat_score(Dep, Score), Score > 0.7.

% Medium threat if score > 0.4
medium_threat(Dep) :- threat_score(Dep, Score), Score > 0.4, Score =< 0.7.

% Transitive threat calculation
transitive_threat(Dep, TotalScore) :-
    dependency(Dep, _, _, _, _),
    findall(Score, (depends_on(Dep, SubDep), threat_score(SubDep, Score)), Scores),
    sum_list(Scores, Sum),
    length(Scores, Count),
    (Count > 0 -> TotalScore is Sum / Count ; TotalScore = 0).

% Query: Find all high-threat dependencies
% ?- high_threat(Dep).
