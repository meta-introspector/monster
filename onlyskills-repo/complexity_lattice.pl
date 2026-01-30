% Lattice of Complexity for Zero Ontology
% Constructs partial order of complexity across languages and implementations

:- module(complexity_lattice, [
    complexity_measure/3,
    lattice_order/2,
    construct_lattice/1,
    visualize_lattice/0
]).

:- dynamic complexity_measure/3.
:- dynamic lattice_order/2.

% Complexity dimensions
complexity_dimension(lines_of_code).
complexity_dimension(cyclomatic).
complexity_dimension(type_complexity).
complexity_dimension(proof_depth).
complexity_dimension(abstraction_level).
complexity_dimension(monster_prime_usage).

% Measure complexity of a file
measure_file_complexity(File, Complexity) :-
    writeln('📊 Measuring File Complexity'),
    format('File: ~w~n', [File]),
    nl,
    
    % Detect language
    detect_language_from_file(File, Lang),
    
    % Measure each dimension
    measure_lines_of_code(File, LOC),
    measure_cyclomatic(File, Lang, Cyclomatic),
    measure_type_complexity(File, Lang, TypeComplexity),
    measure_proof_depth(File, Lang, ProofDepth),
    measure_abstraction(File, Lang, Abstraction),
    measure_monster_prime_usage(File, PrimeUsage),
    
    % Aggregate complexity
    Complexity = complexity(
        file(File),
        language(Lang),
        dimensions([
            lines_of_code(LOC),
            cyclomatic(Cyclomatic),
            type_complexity(TypeComplexity),
            proof_depth(ProofDepth),
            abstraction_level(Abstraction),
            monster_prime_usage(PrimeUsage)
        ]),
        total_score(TotalScore)
    ),
    
    % Calculate total score (weighted sum)
    TotalScore is LOC * 0.1 + Cyclomatic * 0.2 + TypeComplexity * 0.2 + 
                  ProofDepth * 0.3 + Abstraction * 0.1 + PrimeUsage * 0.1,
    
    format('Total complexity score: ~w~n', [TotalScore]),
    
    % Store
    assert(complexity_measure(File, Lang, Complexity)).

% Measure lines of code
measure_lines_of_code(File, LOC) :-
    read_file_to_string(File, Content, []),
    split_string(Content, "\n", "", Lines),
    exclude(is_empty_or_comment, Lines, CodeLines),
    length(CodeLines, LOC).

is_empty_or_comment(Line) :-
    string_trim(Line, Trimmed),
    (   Trimmed = ""
    ;   sub_string(Trimmed, 0, _, _, "%")
    ;   sub_string(Trimmed, 0, _, _, "--")
    ;   sub_string(Trimmed, 0, _, _, "//")
    ).

% Measure cyclomatic complexity
measure_cyclomatic(File, Lang, Cyclomatic) :-
    read_file_to_string(File, Content, []),
    
    % Count decision points
    findall(
        _,
        (
            decision_keyword(Lang, Keyword),
            sub_string(Content, _, _, _, Keyword)
        ),
        Decisions
    ),
    length(Decisions, DecisionCount),
    
    % Cyclomatic = decisions + 1
    Cyclomatic is DecisionCount + 1.

decision_keyword(prolog, ":-").
decision_keyword(prolog, ";").
decision_keyword(lean4, "if").
decision_keyword(lean4, "match").
decision_keyword(coq, "match").
decision_keyword(coq, "if").
decision_keyword(haskell, "case").
decision_keyword(haskell, "if").
decision_keyword(rust, "if").
decision_keyword(rust, "match").

% Measure type complexity
measure_type_complexity(File, Lang, TypeComplexity) :-
    read_file_to_string(File, Content, []),
    
    % Count type definitions
    findall(
        _,
        (
            type_keyword(Lang, Keyword),
            sub_string(Content, _, _, _, Keyword)
        ),
        Types
    ),
    length(Types, TypeCount),
    
    % Count generic/polymorphic types
    findall(
        _,
        (
            generic_marker(Lang, Marker),
            sub_string(Content, _, _, _, Marker)
        ),
        Generics
    ),
    length(Generics, GenericCount),
    
    TypeComplexity is TypeCount + GenericCount * 2.

type_keyword(prolog, ":-").
type_keyword(lean4, "structure").
type_keyword(lean4, "inductive").
type_keyword(coq, "Inductive").
type_keyword(coq, "Record").
type_keyword(haskell, "data").
type_keyword(haskell, "type").
type_keyword(rust, "struct").
type_keyword(rust, "enum").

generic_marker(lean4, "∀").
generic_marker(coq, "forall").
generic_marker(haskell, "=>").
generic_marker(rust, "<").

% Measure proof depth
measure_proof_depth(File, Lang, ProofDepth) :-
    (   proof_language(Lang)
    ->  read_file_to_string(File, Content, []),
        findall(
            _,
            (
                proof_keyword(Lang, Keyword),
                sub_string(Content, _, _, _, Keyword)
            ),
            Proofs
        ),
        length(Proofs, ProofDepth)
    ;   ProofDepth = 0
    ).

proof_language(lean4).
proof_language(coq).
proof_language(agda).

proof_keyword(lean4, "theorem").
proof_keyword(lean4, "lemma").
proof_keyword(coq, "Theorem").
proof_keyword(coq, "Lemma").
proof_keyword(coq, "Proof").
proof_keyword(agda, "≡").

% Measure abstraction level
measure_abstraction(File, Lang, Abstraction) :-
    abstraction_score(Lang, BaseScore),
    
    % Adjust based on file content
    read_file_to_string(File, Content, []),
    
    % Higher abstraction if uses advanced features
    (   sub_string(Content, _, _, _, "meta")
    ->  MetaBonus = 2
    ;   MetaBonus = 0
    ),
    
    (   sub_string(Content, _, _, _, "monad")
    ->  MonadBonus = 1
    ;   MonadBonus = 0
    ),
    
    Abstraction is BaseScore + MetaBonus + MonadBonus.

abstraction_score(prolog, 3).
abstraction_score(lean4, 9).
abstraction_score(agda, 9).
abstraction_score(coq, 8).
abstraction_score(haskell, 7).
abstraction_score(rust, 5).

% Measure Monster prime usage
measure_monster_prime_usage(File, PrimeUsage) :-
    read_file_to_string(File, Content, []),
    
    findall(
        P,
        (
            monster_prime(P),
            number_string(P, PStr),
            sub_string(Content, _, _, _, PStr)
        ),
        UsedPrimes
    ),
    
    length(UsedPrimes, PrimeUsage).

% Construct complexity lattice
construct_lattice(Lattice) :-
    writeln('🔺 Constructing Complexity Lattice'),
    writeln('==================================='),
    nl,
    
    % Measure all files
    findall(File, found_copy(_, File, verified), Files),
    forall(member(File, Files), measure_file_complexity(File, _)),
    
    % Build partial order
    findall(
        File1-File2,
        (
            complexity_measure(File1, _, C1),
            complexity_measure(File2, _, C2),
            less_complex(C1, C2)
        ),
        Orders
    ),
    
    % Store orders
    forall(member(F1-F2, Orders), assert(lattice_order(F1, F2))),
    
    % Build lattice structure
    findall(File, complexity_measure(File, _, _), AllFiles),
    build_lattice_levels(AllFiles, Levels),
    
    Lattice = lattice(
        files(AllFiles),
        orders(Orders),
        levels(Levels)
    ),
    
    format('Lattice constructed: ~w files, ~w orders~n', [length(AllFiles), length(Orders)]).

% Compare complexity
less_complex(C1, C2) :-
    C1 = complexity(_, _, _, total_score(S1)),
    C2 = complexity(_, _, _, total_score(S2)),
    S1 < S2.

% Build lattice levels (topological sort)
build_lattice_levels(Files, Levels) :-
    % Level 0: minimal elements (no predecessors)
    findall(
        File,
        (
            member(File, Files),
            \+ lattice_order(_, File)
        ),
        Level0
    ),
    
    % Build remaining levels
    build_levels([Level0], Files, Levels).

build_levels(Acc, Files, Levels) :-
    Acc = [CurrentLevel|_],
    
    % Find next level (successors of current level)
    findall(
        File,
        (
            member(File, Files),
            \+ member(File, CurrentLevel),
            lattice_order(Pred, File),
            member(Pred, CurrentLevel)
        ),
        NextLevel
    ),
    
    (   NextLevel = []
    ->  reverse(Acc, Levels)
    ;   build_levels([NextLevel|Acc], Files, Levels)
    ).

% Visualize lattice
visualize_lattice :-
    writeln('📊 Complexity Lattice Visualization'),
    writeln('===================================='),
    nl,
    
    construct_lattice(lattice(Files, Orders, Levels)),
    
    % Display levels
    forall(
        nth1(LevelNum, Levels, Level),
        (
            format('Level ~w:~n', [LevelNum]),
            forall(
                member(File, Level),
                (
                    complexity_measure(File, Lang, complexity(_, _, _, total_score(Score))),
                    format('  ~w (~w): ~w~n', [File, Lang, Score])
                )
            ),
            nl
        )
    ),
    
    % Display partial order
    writeln('Partial Order (less complex → more complex):'),
    forall(
        lattice_order(F1, F2),
        (
            complexity_measure(F1, _, complexity(_, _, _, total_score(S1))),
            complexity_measure(F2, _, complexity(_, _, _, total_score(S2))),
            format('  ~w (~w) < ~w (~w)~n', [F1, S1, F2, S2])
        )
    ).

% Query lattice
least_complex(File) :-
    complexity_measure(File, _, complexity(_, _, _, total_score(Score))),
    \+ (
        complexity_measure(_, _, complexity(_, _, _, total_score(S2))),
        S2 < Score
    ).

most_complex(File) :-
    complexity_measure(File, _, complexity(_, _, _, total_score(Score))),
    \+ (
        complexity_measure(_, _, complexity(_, _, _, total_score(S2))),
        S2 > Score
    ).

% Example queries:
% ?- construct_lattice(Lattice).
% ?- visualize_lattice.
% ?- least_complex(File).
% ?- most_complex(File).
% ?- lattice_order(F1, F2).
