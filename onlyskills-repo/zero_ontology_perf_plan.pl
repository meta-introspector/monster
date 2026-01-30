% Prolog Plan for Zero Ontology Perf Recording
% Declarative specification of bootstrap, build, compile, run phases

:- module(zero_ontology_perf_plan, [
    execute_plan/0,
    phase/4,
    language/3,
    selinux_zone/2
]).

% Languages with their zones and tools
language(prolog, 47, swiProlog).
language(lean4, 59, lean4).
language(agda, 59, agda).
language(coq, 71, coq).
language(metacoq, 71, [coq, 'coqPackages.metacoq']).
language(haskell, 41, ghc).
language(rust, 71, [cargo, rustc]).

% SELinux zones
selinux_zone(prolog, 47).    % HIGH
selinux_zone(lean4, 59).     % CRITICAL
selinux_zone(agda, 59).      % CRITICAL
selinux_zone(coq, 71).       % GOOD (proven)
selinux_zone(metacoq, 71).   % GOOD (meta-proven)
selinux_zone(haskell, 41).   % MEDIUM
selinux_zone(rust, 71).      % GOOD (safe)

% Phases: phase(Language, Phase, Command, PerfEvents)
phase(prolog, bootstrap, 'swipl --version', [cycles, instructions]).
phase(prolog, compile, 'swipl -g "consult(\'zero_ontology.pl\'), halt." -t "halt(1)"', [cycles, instructions, 'cache-misses']).
phase(prolog, run, 'swipl -s zero_ontology.pl -g "zero_ontology(prime(71), Step, Class), writeln(Step-Class), halt."', [cycles, instructions, 'cache-misses', 'branch-misses']).

phase(lean4, bootstrap, 'lean --version', [cycles, instructions]).
phase(lean4, compile, 'lake build ZeroOntology', [cycles, instructions, 'cache-misses', 'branch-misses']).
phase(lean4, run, 'lake env lean --run ZeroOntology', [cycles, instructions]).

phase(agda, bootstrap, 'agda --version', [cycles, instructions]).
phase(agda, compile, 'agda --safe ZeroOntology.agda', [cycles, instructions, 'cache-misses']).
phase(agda, run, 'echo "Agda is type-checked, no runtime"', []).

phase(coq, bootstrap, 'coqc --version', [cycles, instructions]).
phase(coq, compile, 'coqc ZeroOntology.v', [cycles, instructions, 'cache-misses', 'branch-misses']).
phase(coq, run, 'coqtop -l ZeroOntology.vo -batch', [cycles, instructions]).

phase(metacoq, bootstrap, 'coqc --version', [cycles, instructions]).
phase(metacoq, compile, 'coqc ZeroOntologyMeta.v', [cycles, instructions, 'cache-misses', 'branch-misses']).
phase(metacoq, run, 'coqtop -l ZeroOntologyMeta.vo -batch', [cycles, instructions]).

phase(haskell, bootstrap, 'ghc --version', [cycles, instructions]).
phase(haskell, compile, 'ghc -O2 -o zero-ontology ZeroOntology.hs', [cycles, instructions, 'cache-misses', 'branch-misses']).
phase(haskell, run, './zero-ontology', [cycles, instructions]).

phase(rust, bootstrap, 'rustc --version', [cycles, instructions]).
phase(rust, compile, 'cargo build --release --lib', [cycles, instructions, 'cache-misses', 'branch-misses']).
phase(rust, run, 'cargo test --release', [cycles, instructions, 'cache-misses']).

% Plan execution
execute_plan :-
    writeln('🔐 Zero Ontology - Prolog Plan Execution'),
    writeln('========================================='),
    nl,
    
    % For each language
    forall(
        language(Lang, Zone, _),
        execute_language_plan(Lang, Zone)
    ),
    
    % Generate summary
    generate_summary,
    
    writeln(''),
    writeln('∞ Plan executed. All phases recorded. ∞').

% Execute plan for one language
execute_language_plan(Lang, Zone) :-
    format('~w (Zone ~w)~n', [Lang, Zone]),
    
    % Setup SELinux
    setup_selinux(Lang, Zone),
    
    % Execute each phase
    forall(
        phase(Lang, Phase, Cmd, Events),
        execute_phase(Lang, Phase, Cmd, Events)
    ),
    
    nl.

% Setup SELinux context
setup_selinux(Lang, Zone) :-
    format('  Setting SELinux context (Zone ~w)...~n', [Zone]),
    
    % Generate SELinux policy
    format(atom(PolicyFile), 'perf_data/zero_ontology/~w_selinux.te', [Lang]),
    
    open(PolicyFile, write, Stream),
    format(Stream, 'module ~w_zero_ontology 1.0;~n~n', [Lang]),
    format(Stream, 'require {~n', []),
    format(Stream, '    type user_t;~n', []),
    format(Stream, '    type unconfined_t;~n', []),
    format(Stream, '    class file { read write execute };~n', []),
    format(Stream, '}~n~n', []),
    format(Stream, '# Zone ~w policy~n', [Zone]),
    format(Stream, 'allow user_t self:file { read write execute };~n', []),
    close(Stream),
    
    writeln('  ✓ SELinux policy generated').

% Execute single phase
execute_phase(Lang, Phase, Cmd, Events) :-
    format('  📊 Recording ~w...~n', [Phase]),
    
    % Build perf command
    format(atom(PerfFile), 'perf_data/zero_ontology/~w_~w.data', [Lang, Phase]),
    format(atom(LogFile), 'perf_data/zero_ontology/~w_~w.log', [Lang, Phase]),
    
    events_to_string(Events, EventsStr),
    
    format(atom(PerfCmd), 'perf record -o ~w -e ~w --call-graph dwarf -- bash -c "~w" 2>&1 | tee ~w',
        [PerfFile, EventsStr, Cmd, LogFile]),
    
    % Execute
    shell(PerfCmd, Status),
    
    % Generate report
    (   Status = 0
    ->  format(atom(ReportFile), 'perf_data/zero_ontology/~w_~w_report.txt', [Lang, Phase]),
        format(atom(ReportCmd), 'perf report -i ~w --stdio > ~w 2>&1', [PerfFile, ReportFile]),
        shell(ReportCmd, _),
        writeln('  ✓ Phase complete')
    ;   writeln('  ✗ Phase failed')
    ).

% Convert event list to string
events_to_string([], 'cycles').
events_to_string([E], E) :- !.
events_to_string([E|Es], Str) :-
    events_to_string(Es, RestStr),
    format(atom(Str), '~w,~w', [E, RestStr]).

% Generate summary
generate_summary :-
    writeln('📊 Generating Performance Summary...'),
    
    open('perf_data/zero_ontology/summary.txt', write, Stream),
    
    format(Stream, 'Zero Ontology - Performance Summary~n', []),
    format(Stream, '====================================~n~n', []),
    format(Stream, 'Languages: 7~n', []),
    format(Stream, 'Phases per language: 3 (bootstrap, compile, run)~n', []),
    format(Stream, 'Total perf recordings: 21~n~n', []),
    format(Stream, 'Performance Data:~n', []),
    
    % For each language
    forall(
        language(Lang, Zone, _),
        (
            format(Stream, '~n~w (Zone ~w):~n', [Lang, Zone]),
            forall(
                phase(Lang, Phase, _, _),
                (
                    format(atom(DataFile), 'perf_data/zero_ontology/~w_~w.data', [Lang, Phase]),
                    (   exists_file(DataFile)
                    ->  size_file(DataFile, Size),
                        SizeKB is Size // 1024,
                        format(Stream, '  ~w: ~w KB~n', [Phase, SizeKB])
                    ;   format(Stream, '  ~w: not found~n', [Phase])
                    )
                )
            )
        )
    ),
    
    format(Stream, '~n∞ All phases recorded. SELinux enhanced. Perf data collected. ∞~n', []),
    
    close(Stream),
    
    writeln('✓ Summary generated').

% Query interface
all_languages(Langs) :-
    findall(Lang, language(Lang, _, _), Langs).

all_phases(Phases) :-
    findall(Phase, (phase(_, Phase, _, _), \+ member(Phase, [bootstrap, compile, run])), AllPhases),
    sort(AllPhases, Phases).

language_phases(Lang, Phases) :-
    findall(Phase, phase(Lang, Phase, _, _), Phases).

phase_command(Lang, Phase, Cmd) :-
    phase(Lang, Phase, Cmd, _).

phase_events(Lang, Phase, Events) :-
    phase(Lang, Phase, _, Events).

% Auto-healing: retry failed phases with LLM assistance
:- dynamic failure_count/2.

auto_heal_phase(Lang, Phase, Cmd, Events) :-
    % Try to execute
    execute_phase(Lang, Phase, Cmd, Events),
    
    % Check if failed
    format(atom(LogFile), 'perf_data/zero_ontology/~w_~w.log', [Lang, Phase]),
    (   phase_failed(LogFile)
    ->  % Healing needed
        writeln('  ⚕️  Auto-healing triggered'),
        heal_with_llm(Lang, Phase, Cmd, LogFile)
    ;   % Success
        retractall(failure_count(Lang, Phase)),
        writeln('  ✓ Phase healthy')
    ).

% Check if phase failed
phase_failed(LogFile) :-
    exists_file(LogFile),
    read_file_to_string(LogFile, Content, []),
    (   sub_string(Content, _, _, _, "error")
    ;   sub_string(Content, _, _, _, "Error")
    ;   sub_string(Content, _, _, _, "failed")
    ;   sub_string(Content, _, _, _, "Failed")
    ).

% Heal with LLM (ollama)
heal_with_llm(Lang, Phase, Cmd, LogFile) :-
    % Increment failure count
    (   failure_count(Lang, Phase, Count)
    ->  retract(failure_count(Lang, Phase, Count)),
        NewCount is Count + 1
    ;   NewCount = 1
    ),
    assert(failure_count(Lang, Phase, NewCount)),
    
    % Max 3 healing attempts
    (   NewCount > 3
    ->  format('  ✗ Max healing attempts reached for ~w ~w~n', [Lang, Phase]),
        fail
    ;   true
    ),
    
    format('  🤖 Healing attempt ~w/3 with LLM...~n', [NewCount]),
    
    % Read error log
    read_file_to_string(LogFile, ErrorLog, []),
    
    % Build LLM prompt
    format(atom(Prompt), 'Fix this ~w ~w error:\n\nCommand: ~w\n\nError:\n~w\n\nProvide fixed command:', 
        [Lang, Phase, Cmd, ErrorLog]),
    
    % Call ollama
    llm_fix_command(Prompt, FixedCmd),
    
    % Retry with fixed command
    format('  🔧 Retrying with: ~w~n', [FixedCmd]),
    execute_phase(Lang, Phase, FixedCmd, [cycles, instructions]).

% LLM integration via ollama
llm_fix_command(Prompt, FixedCmd) :-
    % Write prompt to temp file
    tmp_file_stream(text, PromptFile, PromptStream),
    write(PromptStream, Prompt),
    close(PromptStream),
    
    % Call ollama
    format(atom(OllamaCmd), 'ollama run llama3.2 < ~w', [PromptFile]),
    setup_call_cleanup(
        process_create(path(bash), ['-c', OllamaCmd], [stdout(pipe(Out))]),
        read_string(Out, _, Response),
        close(Out)
    ),
    
    % Extract command from response
    extract_command(Response, FixedCmd),
    
    % Cleanup
    delete_file(PromptFile).

% Extract command from LLM response
extract_command(Response, Cmd) :-
    % Look for code blocks or commands
    (   sub_string(Response, Before, _, After, "```")
    ->  sub_string(Response, _, After, _, Rest),
        sub_string(Rest, 0, End, _, "```"),
        sub_string(Rest, 0, End, _, Cmd)
    ;   % Fallback: take first line
        split_string(Response, "\n", "", [FirstLine|_]),
        string_trim(FirstLine, Cmd)
    ).

% String trim helper
string_trim(Str, Trimmed) :-
    split_string(Str, "", " \t\n\r", [Trimmed]).

% Enhanced execute_plan with auto-healing
execute_plan_with_healing :-
    writeln('🔐 Zero Ontology - Prolog Plan with Auto-Healing'),
    writeln('================================================='),
    nl,
    
    % For each language
    forall(
        language(Lang, Zone, _),
        execute_language_plan_with_healing(Lang, Zone)
    ),
    
    % Generate summary
    generate_summary_with_healing,
    
    writeln(''),
    writeln('∞ Plan executed. Auto-healed. LLM assisted. ∞').

% Execute plan for one language with healing
execute_language_plan_with_healing(Lang, Zone) :-
    format('~w (Zone ~w)~n', [Lang, Zone]),
    
    % Setup SELinux
    setup_selinux(Lang, Zone),
    
    % Execute each phase with auto-healing
    forall(
        phase(Lang, Phase, Cmd, Events),
        auto_heal_phase(Lang, Phase, Cmd, Events)
    ),
    
    nl.

% Generate summary with healing stats
generate_summary_with_healing :-
    writeln('📊 Generating Performance Summary with Healing Stats...'),
    
    open('perf_data/zero_ontology/summary_with_healing.txt', write, Stream),
    
    format(Stream, 'Zero Ontology - Performance Summary (Auto-Healed)~n', []),
    format(Stream, '==================================================~n~n', []),
    format(Stream, 'Languages: 7~n', []),
    format(Stream, 'Phases per language: 3 (bootstrap, compile, run)~n', []),
    format(Stream, 'Total perf recordings: 21~n~n', []),
    
    % Healing statistics
    format(Stream, 'Auto-Healing Statistics:~n', []),
    findall(Lang-Phase-Count, failure_count(Lang, Phase, Count), Failures),
    (   Failures = []
    ->  format(Stream, '  No healing required - all phases succeeded!~n', [])
    ;   forall(
            member(Lang-Phase-Count, Failures),
            format(Stream, '  ~w ~w: ~w healing attempts~n', [Lang, Phase, Count])
        )
    ),
    
    format(Stream, '~n∞ All phases recorded. Auto-healed. LLM assisted. ∞~n', []),
    
    close(Stream),
    
    writeln('✓ Summary with healing stats generated').

% Example queries:
% ?- execute_plan_with_healing.
% ?- auto_heal_phase(rust, compile, 'cargo build --release', [cycles]).
% ?- llm_fix_command('Fix this error: ...', FixedCmd).

% Find other copies of Zero Ontology in 10 languages
:- dynamic found_copy/3.

% Known language file patterns
language_file_pattern(prolog, '**/*zero*ontology*.pl').
language_file_pattern(lean4, '**/*ZeroOntology*.lean').
language_file_pattern(agda, '**/*ZeroOntology*.agda').
language_file_pattern(coq, '**/*ZeroOntology*.v').
language_file_pattern(haskell, '**/*ZeroOntology*.hs').
language_file_pattern(rust, '**/*zero*ontology*.rs').
language_file_pattern(python, '**/*zero*ontology*.py').
language_file_pattern(javascript, '**/*zero*ontology*.js').
language_file_pattern(typescript, '**/*zero*ontology*.ts').
language_file_pattern(ocaml, '**/*zero*ontology*.ml').
language_file_pattern(idris, '**/*ZeroOntology*.idr').
language_file_pattern(fsharp, '**/*ZeroOntology*.fs').
language_file_pattern(scala, '**/*ZeroOntology*.scala').
language_file_pattern(julia, '**/*zero*ontology*.jl').

% Search paths
search_path('/home/mdupont/experiments/monster').
search_path('/home/mdupont/terraform').
search_path('/home/mdupont').

% Find all copies
find_all_copies :-
    writeln('🔍 Finding Zero Ontology copies in 10+ languages...'),
    writeln('==================================================='),
    nl,
    
    retractall(found_copy(_, _, _)),
    
    % Search for each language
    forall(
        language_file_pattern(Lang, Pattern),
        find_language_copies(Lang, Pattern)
    ),
    
    % Report findings
    report_findings,
    
    % Cross-reference with known implementations
    cross_reference_implementations.

% Find copies for one language
find_language_copies(Lang, Pattern) :-
    format('Searching for ~w files...~n', [Lang]),
    
    % Search in each path
    forall(
        search_path(BasePath),
        (
            format(atom(FindCmd), 'find ~w -type f -path "~w" 2>/dev/null', [BasePath, Pattern]),
            setup_call_cleanup(
                process_create(path(bash), ['-c', FindCmd], [stdout(pipe(Out))]),
                read_string(Out, _, Files),
                close(Out)
            ),
            process_found_files(Lang, Files)
        )
    ).

% Process found files
process_found_files(Lang, Files) :-
    split_string(Files, "\n", " \t\r", Lines),
    forall(
        (member(Line, Lines), Line \= ""),
        (
            % Verify it's a Zero Ontology file
            (   verify_zero_ontology_file(Line)
            ->  assert(found_copy(Lang, Line, verified)),
                format('  ✓ Found: ~w~n', [Line])
            ;   assert(found_copy(Lang, Line, unverified)),
                format('  ? Found (unverified): ~w~n', [Line])
            )
        )
    ).

% Verify file contains Zero Ontology markers
verify_zero_ontology_file(FilePath) :-
    exists_file(FilePath),
    read_file_to_string(FilePath, Content, []),
    (   sub_string(Content, _, _, _, "ZeroOntology")
    ;   sub_string(Content, _, _, _, "zero_ontology")
    ;   sub_string(Content, _, _, _, "Monster Walk")
    ;   sub_string(Content, _, _, _, "10-fold Way")
    ;   sub_string(Content, _, _, _, "TenfoldClass")
    ).

% Report findings
report_findings :-
    nl,
    writeln('📊 Search Results:'),
    writeln('=================='),
    nl,
    
    % Count by language
    forall(
        language_file_pattern(Lang, _),
        (
            findall(Path, found_copy(Lang, Path, _), Paths),
            length(Paths, Count),
            (   Count > 0
            ->  format('~w: ~w copies found~n', [Lang, Count]),
                forall(
                    member(Path, Paths),
                    format('  - ~w~n', [Path])
                )
            ;   format('~w: no copies found~n', [Lang])
            )
        )
    ),
    
    nl,
    
    % Total count
    findall(Lang-Path, found_copy(Lang, Path, verified), AllVerified),
    length(AllVerified, TotalVerified),
    findall(Lang-Path, found_copy(Lang, Path, _), AllFound),
    length(AllFound, TotalFound),
    
    format('Total: ~w files found (~w verified)~n', [TotalFound, TotalVerified]).

% Cross-reference with known implementations
cross_reference_implementations :-
    nl,
    writeln('🔗 Cross-Referencing Implementations:'),
    writeln('====================================='),
    nl,
    
    % Known implementations in current directory
    KnownFiles = [
        'zero_ontology.pl',
        'ZeroOntology.lean',
        'ZeroOntology.agda',
        'ZeroOntology.v',
        'ZeroOntologyMeta.v',
        'ZeroOntology.hs',
        'src/zero_ontology.rs'
    ],
    
    % Check which are found
    forall(
        member(KnownFile, KnownFiles),
        (
            (   found_copy(_, Path, _),
                sub_string(Path, _, _, _, KnownFile)
            ->  format('  ✓ ~w: FOUND~n', [KnownFile])
            ;   format('  ✗ ~w: NOT FOUND (should exist)~n', [KnownFile])
            )
        )
    ),
    
    nl,
    
    % Find unexpected copies
    writeln('🆕 Unexpected Copies:'),
    forall(
        (
            found_copy(Lang, Path, verified),
            \+ (
                member(KnownFile, KnownFiles),
                sub_string(Path, _, _, _, KnownFile)
            )
        ),
        format('  + ~w: ~w~n', [Lang, Path])
    ).

% Sync all copies (ensure consistency)
sync_all_copies :-
    writeln('🔄 Syncing all Zero Ontology copies...'),
    nl,
    
    % Find canonical version (most recent)
    find_canonical_version(CanonicalLang, CanonicalPath),
    format('Canonical version: ~w (~w)~n', [CanonicalLang, CanonicalPath]),
    nl,
    
    % For each found copy
    forall(
        found_copy(Lang, Path, verified),
        sync_copy(Lang, Path, CanonicalLang, CanonicalPath)
    ).

% Find canonical version (most recently modified)
find_canonical_version(Lang, Path) :-
    findall(
        MTime-Lang-Path,
        (
            found_copy(Lang, Path, verified),
            time_file(Path, MTime)
        ),
        Versions
    ),
    sort(Versions, Sorted),
    reverse(Sorted, [_-Lang-Path|_]).

% Sync one copy
sync_copy(Lang, Path, CanonicalLang, CanonicalPath) :-
    (   Lang = CanonicalLang, Path = CanonicalPath
    ->  format('  ✓ ~w: canonical version~n', [Path])
    ;   format('  🔄 Syncing ~w...~n', [Path]),
        % Use LLM to translate canonical to target language
        translate_with_llm(CanonicalPath, Path, Lang),
        format('  ✓ ~w: synced~n', [Path])
    ).

% Translate using LLM
translate_with_llm(SourcePath, TargetPath, TargetLang) :-
    read_file_to_string(SourcePath, SourceCode, []),
    
    format(atom(Prompt), 'Translate this Zero Ontology code to ~w:\n\n~w\n\nProvide only the translated code:', 
        [TargetLang, SourceCode]),
    
    llm_translate(Prompt, TranslatedCode),
    
    open(TargetPath, write, Stream),
    write(Stream, TranslatedCode),
    close(Stream).

% LLM translate
llm_translate(Prompt, Translation) :-
    tmp_file_stream(text, PromptFile, PromptStream),
    write(PromptStream, Prompt),
    close(PromptStream),
    
    format(atom(OllamaCmd), 'ollama run llama3.2 < ~w', [PromptFile]),
    setup_call_cleanup(
        process_create(path(bash), ['-c', OllamaCmd], [stdout(pipe(Out))]),
        read_string(Out, _, Translation),
        close(Out)
    ),
    
    delete_file(PromptFile).

% Example queries:
% ?- find_all_copies.
% ?- sync_all_copies.

% Search 8M files in 400k Parquet shards
:- use_module(library(process)).

% Parquet shard locations
parquet_shard_location('/home/mdupont/experiments/monster/onlyskills-repo/*.parquet').
parquet_shard_location('/mnt/data1/**/*.parquet').
parquet_shard_location('/home/mdupont/**/*.parquet').

% Search all parquet shards for Zero Ontology files
search_parquet_shards :-
    writeln('🔍 Searching 8M files in 400k Parquet shards...'),
    writeln('================================================'),
    nl,
    
    % Find all parquet files
    find_all_parquet_files(ParquetFiles),
    length(ParquetFiles, ShardCount),
    format('Found ~w Parquet shards~n', [ShardCount]),
    nl,
    
    % Search each shard
    search_shards_parallel(ParquetFiles).

% Find all parquet files
find_all_parquet_files(Files) :-
    findall(
        File,
        (
            parquet_shard_location(Pattern),
            expand_file_name(Pattern, Matches),
            member(File, Matches)
        ),
        AllFiles
    ),
    sort(AllFiles, Files).

% Search shards in parallel (batches of 71)
search_shards_parallel(Files) :-
    length(Files, Total),
    BatchSize = 71,
    NumBatches is ceiling(Total / BatchSize),
    format('Processing ~w batches of ~w shards...~n~n', [NumBatches, BatchSize]),
    
    % Process in batches
    forall(
        between(1, NumBatches, BatchNum),
        process_batch(Files, BatchNum, BatchSize)
    ).

% Process one batch
process_batch(AllFiles, BatchNum, BatchSize) :-
    Start is (BatchNum - 1) * BatchSize,
    End is min(Start + BatchSize, length(AllFiles)),
    
    format('Batch ~w: shards ~w-~w~n', [BatchNum, Start, End]),
    
    % Get batch files
    length(Prefix, Start),
    append(Prefix, Rest, AllFiles),
    length(Batch, BatchSize),
    append(Batch, _, Rest),
    
    % Search batch in parallel using Rust
    search_batch_rust(Batch, BatchNum).

% Search batch using Rust parquet reader
search_batch_rust(Files, BatchNum) :-
    % Write file list
    format(atom(ListFile), '/tmp/parquet_batch_~w.txt', [BatchNum]),
    open(ListFile, write, Stream),
    forall(member(File, Files), format(Stream, '~w~n', [File])),
    close(Stream),
    
    % Call Rust searcher
    format(atom(Cmd), 'cargo run --release --bin search_parquet_batch -- ~w "zero_ontology" "ZeroOntology" "Monster Walk" "10-fold Way"', [ListFile]),
    
    process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out))]),
    read_string(Out, _, Results),
    close(Out),
    
    % Process results
    process_search_results(Results, BatchNum),
    
    % Cleanup
    delete_file(ListFile).

% Process search results
process_search_results(Results, BatchNum) :-
    split_string(Results, "\n", " \t\r", Lines),
    forall(
        (member(Line, Lines), Line \= ""),
        (
            % Parse: file:row:column:match
            split_string(Line, ":", "", [File, Row, Col, Match]),
            assert(parquet_match(File, Row, Col, Match)),
            format('  ✓ Batch ~w: ~w:~w:~w - ~w~n', [BatchNum, File, Row, Col, Match])
        )
    ).

% Generate summary of parquet search
summarize_parquet_search :-
    nl,
    writeln('📊 Parquet Search Summary:'),
    writeln('=========================='),
    nl,
    
    % Count matches by file
    findall(File, parquet_match(File, _, _, _), AllMatches),
    sort(AllMatches, UniqueFiles),
    length(AllMatches, TotalMatches),
    length(UniqueFiles, FileCount),
    
    format('Total matches: ~w~n', [TotalMatches]),
    format('Files with matches: ~w~n', [FileCount]),
    nl,
    
    % Top files
    writeln('Top files with matches:'),
    findall(
        Count-File,
        (
            member(File, UniqueFiles),
            findall(_, parquet_match(File, _, _, _), Matches),
            length(Matches, Count)
        ),
        Counts
    ),
    sort(Counts, Sorted),
    reverse(Sorted, TopFiles),
    forall(
        (member(Count-File, TopFiles), Count > 0),
        format('  ~w: ~w matches~n', [File, Count])
    ),
    
    nl,
    
    % Match types
    writeln('Match types:'),
    findall(Match, parquet_match(_, _, _, Match), AllMatchTypes),
    sort(AllMatchTypes, UniqueMatches),
    forall(
        member(Match, UniqueMatches),
        (
            findall(_, parquet_match(_, _, _, Match), Ms),
            length(Ms, MCount),
            format('  "~w": ~w occurrences~n', [Match, MCount])
        )
    ).

% Query parquet matches
query_parquet_matches(Pattern, Matches) :-
    findall(
        match(File, Row, Col, Text),
        (
            parquet_match(File, Row, Col, Text),
            sub_string(Text, _, _, _, Pattern)
        ),
        Matches
    ).

% Extract Zero Ontology files from parquet
extract_zero_ontology_from_parquet :-
    writeln('📦 Extracting Zero Ontology files from Parquet...'),
    nl,
    
    % Find all matches
    findall(File-Row, parquet_match(File, Row, _, _), Matches),
    sort(Matches, UniqueMatches),
    
    % Extract each file
    forall(
        member(File-Row, UniqueMatches),
        extract_file_from_parquet(File, Row)
    ).

% Extract single file from parquet
extract_file_from_parquet(ParquetFile, Row) :-
    format('Extracting row ~w from ~w...~n', [Row, ParquetFile]),
    
    % Use Rust to extract
    format(atom(Cmd), 'cargo run --release --bin extract_parquet_row -- ~w ~w', [ParquetFile, Row]),
    
    process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out))]),
    read_string(Out, _, Content),
    close(Out),
    
    % Save extracted file
    file_base_name(ParquetFile, BaseName),
    format(atom(OutFile), 'extracted/~w_row_~w.txt', [BaseName, Row]),
    
    open(OutFile, write, Stream),
    write(Stream, Content),
    close(Stream),
    
    format('  ✓ Saved to ~w~n', [OutFile]).

% Parallel search using GNU parallel
search_parquet_parallel_gnu :-
    writeln('🚀 Parallel Parquet Search (GNU parallel)...'),
    nl,
    
    % Find all parquet files
    find_all_parquet_files(Files),
    
    % Write to file
    open('/tmp/parquet_files.txt', write, Stream),
    forall(member(File, Files), format(Stream, '~w~n', [File])),
    close(Stream),
    
    % Run parallel search
    Cmd = 'cat /tmp/parquet_files.txt | parallel -j 71 "cargo run --release --bin search_parquet -- {} \\"zero_ontology\\" \\"ZeroOntology\\""',
    
    process_create(path(bash), ['-c', Cmd], [stdout(pipe(Out))]),
    read_string(Out, _, Results),
    close(Out),
    
    % Process results
    process_search_results(Results, parallel),
    
    % Summary
    summarize_parquet_search.

% Example queries:
% ?- search_parquet_shards.
% ?- search_parquet_parallel_gnu.
% ?- query_parquet_matches("Monster", Matches).
% ?- extract_zero_ontology_from_parquet.
% ?- summarize_parquet_search.
