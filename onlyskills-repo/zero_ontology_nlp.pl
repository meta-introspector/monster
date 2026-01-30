% NLP and Introspection for Zero Ontology Search
% Apply natural language processing and meta-reasoning

:- module(zero_ontology_nlp, [
    nlp_analyze/2,
    introspect_search/1,
    semantic_similarity/3,
    extract_concepts/2
]).

% NLP Analysis of search results
nlp_analyze(SearchResults, Analysis) :-
    writeln('🧠 NLP Analysis of Search Results'),
    writeln('=================================='),
    nl,
    
    % Extract text from results
    extract_all_text(SearchResults, AllText),
    
    % Tokenize
    tokenize_text(AllText, Tokens),
    format('Tokens: ~w~n', [length(Tokens)]),
    
    % Extract concepts
    extract_concepts(Tokens, Concepts),
    format('Concepts: ~w~n', [Concepts]),
    nl,
    
    % Named entity recognition
    extract_entities(Tokens, Entities),
    format('Entities: ~w~n', [Entities]),
    nl,
    
    % Semantic clustering
    cluster_by_semantics(Concepts, Clusters),
    format('Semantic clusters: ~w~n', [length(Clusters)]),
    
    % Build analysis
    Analysis = analysis(
        tokens(Tokens),
        concepts(Concepts),
        entities(Entities),
        clusters(Clusters)
    ).

% Extract all text from search results
extract_all_text(Results, AllText) :-
    findall(
        Text,
        (
            member(match(_, _, _, Text), Results)
        ),
        Texts
    ),
    atomic_list_concat(Texts, ' ', AllText).

% Tokenize text
tokenize_text(Text, Tokens) :-
    split_string(Text, " \t\n\r.,;:!?()[]{}\"'", "", RawTokens),
    maplist(string_lower, RawTokens, LowerTokens),
    exclude(=(""), LowerTokens, Tokens).

% Extract concepts (nouns, technical terms)
extract_concepts(Tokens, Concepts) :-
    % Technical terms related to Zero Ontology
    TechnicalTerms = [
        "monster", "walk", "tenfold", "genus", "prime",
        "ontology", "zero", "symmetry", "topology", "group",
        "lean", "coq", "agda", "prolog", "haskell", "rust"
    ],
    
    findall(
        Concept,
        (
            member(Token, Tokens),
            member(Term, TechnicalTerms),
            sub_string(Token, _, _, _, Term),
            Concept = Token
        ),
        AllConcepts
    ),
    sort(AllConcepts, Concepts).

% Named entity recognition
extract_entities(Tokens, Entities) :-
    % Recognize Monster primes
    MonsterPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71],
    
    findall(
        prime(P),
        (
            member(Token, Tokens),
            atom_number(Token, N),
            member(N, MonsterPrimes),
            P = N
        ),
        PrimeEntities
    ),
    
    % Recognize languages
    Languages = [prolog, lean4, agda, coq, haskell, rust, python],
    
    findall(
        language(Lang),
        (
            member(Token, Tokens),
            member(Lang, Languages),
            atom_string(Lang, LangStr),
            sub_string(Token, _, _, _, LangStr)
        ),
        LangEntities
    ),
    
    append(PrimeEntities, LangEntities, Entities).

% Semantic clustering
cluster_by_semantics(Concepts, Clusters) :-
    % Cluster by semantic similarity
    findall(
        cluster(Theme, Members),
        (
            semantic_theme(Theme),
            findall(
                Concept,
                (
                    member(Concept, Concepts),
                    concept_matches_theme(Concept, Theme)
                ),
                Members
            ),
            Members \= []
        ),
        Clusters
    ).

% Semantic themes
semantic_theme(mathematics).
semantic_theme(programming).
semantic_theme(topology).
semantic_theme(group_theory).
semantic_theme(proof_systems).

% Concept matches theme
concept_matches_theme(Concept, mathematics) :-
    member(Term, ["prime", "genus", "group", "symmetry"]),
    sub_string(Concept, _, _, _, Term).

concept_matches_theme(Concept, programming) :-
    member(Term, ["rust", "haskell", "prolog", "lean", "code"]),
    sub_string(Concept, _, _, _, Term).

concept_matches_theme(Concept, topology) :-
    member(Term, ["genus", "topology", "manifold", "surface"]),
    sub_string(Concept, _, _, _, Term).

concept_matches_theme(Concept, group_theory) :-
    member(Term, ["monster", "group", "symmetry", "walk"]),
    sub_string(Concept, _, _, _, Term).

concept_matches_theme(Concept, proof_systems) :-
    member(Term, ["lean", "coq", "agda", "proof", "theorem"]),
    sub_string(Concept, _, _, _, Term).

% Semantic similarity using word embeddings (via LLM)
semantic_similarity(Text1, Text2, Similarity) :-
    % Use LLM to compute semantic similarity
    format(atom(Prompt), 'Rate semantic similarity (0-1) between:\n1: ~w\n2: ~w\nProvide only the number:', 
        [Text1, Text2]),
    
    llm_query(Prompt, Response),
    atom_number(Response, Similarity).

% LLM query
llm_query(Prompt, Response) :-
    tmp_file_stream(text, PromptFile, PromptStream),
    write(PromptStream, Prompt),
    close(PromptStream),
    
    format(atom(Cmd), 'ollama run llama3.2 < ~w', [PromptFile]),
    setup_call_cleanup(
        process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out))]),
        read_string(Out, _, RawResponse),
        close(Out)
    ),
    
    string_trim(RawResponse, Response),
    delete_file(PromptFile).

% Introspect search process
introspect_search(Introspection) :-
    writeln('🔍 Introspecting Search Process'),
    writeln('================================'),
    nl,
    
    % Analyze search strategy
    current_predicate(search_parquet_shards/0),
    predicate_property(search_parquet_shards, Properties),
    format('Search predicate properties: ~w~n', [Properties]),
    nl,
    
    % Count search operations
    findall(Op, search_operation(Op), Operations),
    length(Operations, OpCount),
    format('Search operations: ~w~n', [OpCount]),
    nl,
    
    % Analyze performance
    findall(Time-Op, search_timing(Op, Time), Timings),
    (   Timings \= []
    ->  maplist(arg(1), Timings, Times),
        sumlist(Times, TotalTime),
        length(Timings, N),
        AvgTime is TotalTime / N,
        format('Average search time: ~w ms~n', [AvgTime])
    ;   writeln('No timing data available')
    ),
    nl,
    
    % Meta-reasoning about search
    meta_reason_search(MetaReasoning),
    format('Meta-reasoning: ~w~n', [MetaReasoning]),
    nl,
    
    % Build introspection
    Introspection = introspection(
        properties(Properties),
        operations(Operations),
        timings(Timings),
        meta_reasoning(MetaReasoning)
    ).

% Search operations
search_operation(find_parquet_files).
search_operation(search_batch).
search_operation(process_results).
search_operation(extract_matches).

% Search timing (dynamic)
:- dynamic search_timing/2.

% Meta-reasoning about search
meta_reason_search(Reasoning) :-
    % Analyze search effectiveness
    findall(_, parquet_match(_, _, _, _), Matches),
    length(Matches, MatchCount),
    
    % Analyze search coverage
    findall(_, found_copy(_, _, _), Copies),
    length(Copies, CopyCount),
    
    % Reason about results
    (   MatchCount > 100
    ->  Effectiveness = high
    ;   MatchCount > 10
    ->  Effectiveness = medium
    ;   Effectiveness = low
    ),
    
    (   CopyCount >= 7
    ->  Coverage = complete
    ;   CopyCount >= 4
    ->  Coverage = partial
    ;   Coverage = incomplete
    ),
    
    Reasoning = reasoning(
        effectiveness(Effectiveness),
        coverage(Coverage),
        matches(MatchCount),
        copies(CopyCount),
        recommendation(Recommendation)
    ),
    
    % Generate recommendation
    recommend_action(Effectiveness, Coverage, Recommendation).

% Recommend action based on meta-reasoning
recommend_action(high, complete, 'Search is optimal. Continue monitoring.').
recommend_action(high, partial, 'Search is effective but coverage incomplete. Expand search paths.').
recommend_action(medium, complete, 'Search coverage is good. Optimize search patterns.').
recommend_action(medium, partial, 'Moderate results. Refine search strategy and expand paths.').
recommend_action(low, _, 'Low effectiveness. Review search patterns and add more sources.').

% NLP-enhanced search query
nlp_search_query(NaturalQuery, StructuredQuery) :-
    writeln('🗣️  NLP-Enhanced Search Query'),
    nl,
    
    % Parse natural language query
    format('Natural query: ~w~n', [NaturalQuery]),
    
    % Extract intent
    extract_intent(NaturalQuery, Intent),
    format('Intent: ~w~n', [Intent]),
    
    % Extract entities from query
    tokenize_text(NaturalQuery, QueryTokens),
    extract_entities(QueryTokens, QueryEntities),
    format('Entities: ~w~n', [QueryEntities]),
    
    % Build structured query
    build_structured_query(Intent, QueryEntities, StructuredQuery),
    format('Structured query: ~w~n', [StructuredQuery]).

% Extract intent from natural language
extract_intent(Query, Intent) :-
    (   sub_string(Query, _, _, _, "find")
    ->  Intent = search
    ;   sub_string(Query, _, _, _, "show")
    ->  Intent = display
    ;   sub_string(Query, _, _, _, "analyze")
    ->  Intent = analyze
    ;   sub_string(Query, _, _, _, "compare")
    ->  Intent = compare
    ;   Intent = unknown
    ).

% Build structured query
build_structured_query(search, Entities, query(search, Entities, [])).
build_structured_query(display, Entities, query(display, Entities, [])).
build_structured_query(analyze, Entities, query(analyze, Entities, [nlp, introspect])).
build_structured_query(compare, Entities, query(compare, Entities, [similarity])).

% Example queries:
% ?- nlp_analyze([match(file, 1, col, "Monster Walk")], Analysis).
% ?- introspect_search(Intro).
% ?- semantic_similarity("Monster Walk", "10-fold Way", Sim).
% ?- nlp_search_query("Find all Zero Ontology files in Rust", Query).

% Monster Prime Analysis with p-adic and English names
:- use_module(library(clpfd)).

% Monster primes
monster_prime(2).
monster_prime(3).
monster_prime(5).
monster_prime(7).
monster_prime(11).
monster_prime(13).
monster_prime(17).
monster_prime(19).
monster_prime(23).
monster_prime(29).
monster_prime(31).
monster_prime(41).
monster_prime(47).
monster_prime(59).
monster_prime(71).

% English names for numbers
english_name(0, zero).
english_name(1, one).
english_name(2, two).
english_name(3, three).
english_name(4, four).
english_name(5, five).
english_name(6, six).
english_name(7, seven).
english_name(8, eight).
english_name(9, nine).
english_name(10, ten).
english_name(11, eleven).
english_name(12, twelve).
english_name(13, thirteen).
english_name(14, fourteen).
english_name(15, fifteen).
english_name(16, sixteen).
english_name(17, seventeen).
english_name(18, eighteen).
english_name(19, nineteen).
english_name(20, twenty).
english_name(23, 'twenty-three').
english_name(29, 'twenty-nine').
english_name(31, 'thirty-one').
english_name(41, 'forty-one').
english_name(47, 'forty-seven').
english_name(59, 'fifty-nine').
english_name(71, 'seventy-one').

% p-adic valuation: highest power of p dividing n
padic_valuation(N, P, V) :-
    padic_val(N, P, 0, V).

padic_val(N, P, Acc, V) :-
    (   N mod P =:= 0
    ->  N1 is N // P,
        Acc1 is Acc + 1,
        padic_val(N1, P, Acc1, V)
    ;   V = Acc
    ).

% p-adic norm: |n|_p = p^(-v_p(n))
padic_norm(N, P, Norm) :-
    padic_valuation(N, P, V),
    Norm is P ** (-V).

% Analyze number by all Monster primes
analyze_by_monster_primes(N, Analysis) :-
    writeln('🔢 Monster Prime Analysis'),
    format('Number: ~w~n', [N]),
    
    % English name
    (   english_name(N, Name)
    ->  format('English: ~w~n', [Name])
    ;   format('English: (no name)~n', [])
    ),
    nl,
    
    % Divisibility by each Monster prime
    writeln('Divisibility:'),
    findall(
        P-Div,
        (
            monster_prime(P),
            (   N mod P =:= 0
            ->  Div = yes
            ;   Div = no
            ),
            english_name(P, PName),
            format('  ~w (~w): ~w~n', [P, PName, Div])
        ),
        Divisibility
    ),
    nl,
    
    % p-adic valuations
    writeln('p-adic valuations:'),
    findall(
        P-V,
        (
            monster_prime(P),
            padic_valuation(N, P, V),
            V > 0,
            english_name(P, PName),
            format('  v_~w(~w) = ~w  (divides ~w^~w times)~n', [P, N, V, PName, V])
        ),
        Valuations
    ),
    nl,
    
    % p-adic norms
    writeln('p-adic norms:'),
    findall(
        P-Norm,
        (
            monster_prime(P),
            padic_norm(N, P, Norm),
            english_name(P, PName),
            format('  |~w|_~w = ~w~n', [N, P, Norm])
        ),
        Norms
    ),
    nl,
    
    % Prime factorization
    prime_factorization(N, Factors),
    format('Prime factorization: ~w~n', [Factors]),
    
    % Express in English
    factorization_english(Factors, English),
    format('In English: ~w~n', [English]),
    nl,
    
    Analysis = analysis(
        number(N),
        english(Name),
        divisibility(Divisibility),
        padic_valuations(Valuations),
        padic_norms(Norms),
        factorization(Factors),
        factorization_english(English)
    ).

% Prime factorization
prime_factorization(N, Factors) :-
    factorize(N, [], Factors).

factorize(1, Acc, Factors) :- !, reverse(Acc, Factors).
factorize(N, Acc, Factors) :-
    smallest_prime_factor(N, P),
    padic_valuation(N, P, V),
    N1 is N // (P ** V),
    factorize(N1, [P^V|Acc], Factors).

smallest_prime_factor(N, P) :-
    between(2, N, P),
    N mod P =:= 0,
    is_prime(P),
    !.

is_prime(2) :- !.
is_prime(N) :-
    N > 2,
    N mod 2 =\= 0,
    \+ has_factor(N, 3).

has_factor(N, F) :-
    F * F =< N,
    (   N mod F =:= 0
    ->  true
    ;   F2 is F + 2,
        has_factor(N, F2)
    ).

% Express factorization in English
factorization_english([], 'one').
factorization_english([P^1], Name) :-
    !,
    english_name(P, Name).
factorization_english([P^E], Result) :-
    !,
    english_name(P, PName),
    english_name(E, EName),
    format(atom(Result), '~w to the power of ~w', [PName, EName]).
factorization_english([P^E|Rest], Result) :-
    english_name(P, PName),
    english_name(E, EName),
    factorization_english(Rest, RestEnglish),
    format(atom(Result), '~w to the power of ~w times ~w', [PName, EName, RestEnglish]).

% NLP search with Monster prime filtering
nlp_search_with_primes(Query, PrimeFilter, Results) :-
    writeln('🔍 NLP Search with Monster Prime Filter'),
    format('Query: ~w~n', [Query]),
    format('Prime filter: ~w~n', [PrimeFilter]),
    nl,
    
    % Parse query
    nlp_search_query(Query, StructuredQuery),
    
    % Apply prime filter
    findall(
        Match,
        (
            parquet_match(File, Row, Col, Text),
            % Extract numbers from text
            extract_numbers(Text, Numbers),
            % Check if any number satisfies prime filter
            member(N, Numbers),
            satisfies_prime_filter(N, PrimeFilter)
        ),
        Results
    ),
    
    length(Results, Count),
    format('Found ~w results~n', [Count]).

% Extract numbers from text
extract_numbers(Text, Numbers) :-
    split_string(Text, " \t\n\r.,;:!?()[]{}\"'", "", Tokens),
    findall(
        N,
        (
            member(Token, Tokens),
            atom_number(Token, N)
        ),
        Numbers
    ).

% Check if number satisfies prime filter
satisfies_prime_filter(N, divisible_by(P)) :-
    monster_prime(P),
    N mod P =:= 0.

satisfies_prime_filter(N, padic_val_gt(P, V)) :-
    monster_prime(P),
    padic_valuation(N, P, Val),
    Val > V.

satisfies_prime_filter(N, is_monster_prime) :-
    monster_prime(N).

satisfies_prime_filter(N, english_contains(Word)) :-
    english_name(N, Name),
    atom_string(Name, NameStr),
    atom_string(Word, WordStr),
    sub_string(NameStr, _, _, _, WordStr).

% Analyze text by Monster primes
analyze_text_by_primes(Text, Analysis) :-
    writeln('📊 Text Analysis by Monster Primes'),
    nl,
    
    % Extract numbers
    extract_numbers(Text, Numbers),
    format('Numbers found: ~w~n', [Numbers]),
    nl,
    
    % Analyze each number
    findall(
        N-A,
        (
            member(N, Numbers),
            analyze_by_monster_primes(N, A)
        ),
        Analyses
    ),
    
    % Aggregate statistics
    findall(P, (member(_-analysis(_, _, Div, _, _, _, _), Analyses), member(P-yes, Div)), AllDivisible),
    sort(AllDivisible, UniqueDivisible),
    
    format('Numbers divisible by Monster primes: ~w~n', [UniqueDivisible]),
    
    Analysis = text_analysis(
        numbers(Numbers),
        analyses(Analyses),
        divisible_by(UniqueDivisible)
    ).

% English name search
search_by_english_name(EnglishName, Numbers) :-
    findall(
        N,
        (
            english_name(N, Name),
            atom_string(Name, NameStr),
            atom_string(EnglishName, QueryStr),
            sub_string(NameStr, _, _, _, QueryStr)
        ),
        Numbers
    ).

% Example queries:
% ?- analyze_by_monster_primes(71, Analysis).
% ?- padic_valuation(8080, 2, V).
% ?- padic_norm(71, 71, Norm).
% ?- nlp_search_with_primes("Find files", divisible_by(71), Results).
% ?- analyze_text_by_primes("The Monster has order 71 and genus 0", Analysis).
% ?- search_by_english_name("seven", Numbers).


% Native Vernacular Introspection for each found file
% Runs language-specific introspection tools

% Language-specific introspection commands
introspection_command(prolog, File, Cmd) :-
    format(atom(Cmd), 'swipl -s ~w -g "listing, halt."', [File]).

introspection_command(lean4, File, Cmd) :-
    format(atom(Cmd), 'lean --server < ~w', [File]).

introspection_command(agda, File, Cmd) :-
    format(atom(Cmd), 'agda --show-implicit --show-irrelevant ~w', [File]).

introspection_command(coq, File, Cmd) :-
    format(atom(Cmd), 'coqtop -l ~w -batch -quiet', [File]).

introspection_command(haskell, File, Cmd) :-
    format(atom(Cmd), 'ghci ~w -e ":browse" -e ":info"', [File]).

introspection_command(rust, File, Cmd) :-
    format(atom(Cmd), 'cargo expand --manifest-path ~w', [File]).

% Detect language from file extension
detect_language_from_file(File, Lang) :-
    file_name_extension(_, Ext, File),
    extension_language(Ext, Lang).

extension_language(pl, prolog).
extension_language(lean, lean4).
extension_language(agda, agda).
extension_language(v, coq).
extension_language(hs, haskell).
extension_language(rs, rust).

% Run native introspection on file
introspect_file_native(File, Introspection) :-
    writeln('🔬 Native Vernacular Introspection'),
    format('File: ~w~n', [File]),
    nl,
    
    % Detect language
    detect_language_from_file(File, Lang),
    format('Language: ~w~n', [Lang]),
    
    % Get introspection command
    introspection_command(Lang, File, Cmd),
    format('Command: ~w~n', [Cmd]),
    nl,
    
    % Run introspection
    writeln('Running introspection...'),
    setup_call_cleanup(
        process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out)), stderr(pipe(Err))]),
        (
            read_string(Out, _, Output),
            read_string(Err, _, Errors)
        ),
        (close(Out), close(Err))
    ),
    
    Introspection = introspection(
        file(File),
        language(Lang),
        command(Cmd),
        output(Output),
        errors(Errors)
    ).

% Introspect all found files
introspect_all_found_files :-
    writeln('🔬 Introspecting All Found Files'),
    writeln('================================='),
    nl,
    
    % Get all found files
    findall(File, found_copy(_, File, verified), Files),
    length(Files, Count),
    format('Found ~w files to introspect~n~n', [Count]),
    
    % Introspect each file
    forall(
        member(File, Files),
        (
            introspect_file_native(File, Introspection),
            assert(file_introspection(File, Introspection)),
            nl
        )
    ).

% Example queries:
% ?- introspect_file_native('zero_ontology.pl', Intro).
% ?- introspect_all_found_files.


% Native Vernacular Introspection for each found file
% Runs language-specific introspection tools

:- dynamic file_introspection/2.

% Language-specific introspection commands
introspection_command(prolog, File, Cmd) :-
    format(atom(Cmd), 'swipl -s ~w -g "listing, halt."', [File]).

introspection_command(lean4, File, Cmd) :-
    format(atom(Cmd), 'lean --server < ~w', [File]).

introspection_command(agda, File, Cmd) :-
    format(atom(Cmd), 'agda --show-implicit --show-irrelevant ~w', [File]).

introspection_command(coq, File, Cmd) :-
    format(atom(Cmd), 'coqtop -l ~w -batch -quiet', [File]).

introspection_command(haskell, File, Cmd) :-
    format(atom(Cmd), 'ghci ~w -e ":browse" -e ":info"', [File]).

introspection_command(rust, File, Cmd) :-
    format(atom(Cmd), 'cargo expand --manifest-path ~w', [File]).

% Detect language from file extension
detect_language_from_file(File, Lang) :-
    file_name_extension(_, Ext, File),
    extension_language(Ext, Lang).

extension_language(pl, prolog).
extension_language(lean, lean4).
extension_language(agda, agda).
extension_language(v, coq).
extension_language(hs, haskell).
extension_language(rs, rust).

% Run native introspection on file
introspect_file_native(File, Introspection) :-
    writeln('🔬 Native Vernacular Introspection'),
    format('File: ~w~n', [File]),
    nl,
    
    % Detect language
    detect_language_from_file(File, Lang),
    format('Language: ~w~n', [Lang]),
    
    % Get introspection command
    introspection_command(Lang, File, Cmd),
    format('Command: ~w~n', [Cmd]),
    nl,
    
    % Run introspection
    writeln('Running introspection...'),
    setup_call_cleanup(
        process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out)), stderr(pipe(Err))]),
        (
            read_string(Out, _, Output),
            read_string(Err, _, Errors)
        ),
        (close(Out), close(Err))
    ),
    
    Introspection = introspection(
        file(File),
        language(Lang),
        command(Cmd),
        output(Output),
        errors(Errors)
    ),
    
    format('✓ Introspection complete (~w bytes output)~n', [string_length(Output)]).

% Introspect all found files
introspect_all_found_files :-
    writeln('🔬 Introspecting All Found Files'),
    writeln('================================='),
    nl,
    
    % Get all found files
    findall(File, found_copy(_, File, verified), Files),
    length(Files, Count),
    format('Found ~w files to introspect~n~n', [Count]),
    
    % Introspect each file
    forall(
        member(File, Files),
        (
            introspect_file_native(File, Introspection),
            assert(file_introspection(File, Introspection)),
            nl
        )
    ),
    
    writeln('∞ All files introspected. Native vernacular complete. ∞').

% Example queries:
% ?- introspect_file_native('zero_ontology.pl', Intro).
% ?- introspect_all_found_files.
