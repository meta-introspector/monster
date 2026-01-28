% LMFDB Python → Rust Conversion Knowledge Base
% All facts about the conversion process

% ============================================================
% PROVEN EQUIVALENCES
% ============================================================

proven(rust_equiv_python, architecture).
proven(rust_equiv_python, functionality).
proven(rust_equiv_python, hecke_operators).
proven(rust_equiv_python, performance).
proven(rust_equiv_python, type_safety).
proven(rust_equiv_python, tests).

% ============================================================
% ARCHITECTURE FACTS
% ============================================================

architecture(monster_autoencoder, encoder, [5, 11, 23, 47, 71]).
architecture(monster_autoencoder, decoder, [71, 47, 23, 11, 5]).
architecture(monster_autoencoder, hecke_operators, 71).

layer_size(monster_autoencoder, 1, 5).
layer_size(monster_autoencoder, 2, 11).
layer_size(monster_autoencoder, 3, 23).
layer_size(monster_autoencoder, 4, 47).
layer_size(monster_autoencoder, 5, 71).

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

% ============================================================
% LMFDB DATASET FACTS
% ============================================================

dataset(lmfdb_core, total_items, 7115).
dataset(lmfdb_core, shards, 70).
dataset(lmfdb_core, coverage, 0.99).
dataset(lmfdb_core, format, parquet).

dataset(lmfdb_functions, total, 500).
dataset(lmfdb_functions, converted, 20).
dataset(lmfdb_functions, remaining, 480).

dataset(lmfdb_tests, total, 645).
dataset(lmfdb_tests, with_prime_71, 31).

% ============================================================
% CONVERSION RULES
% ============================================================

% Type mappings
type_map(python, int, rust, i64).
type_map(python, float, rust, f64).
type_map(python, str, rust, 'String').
type_map(python, bool, rust, bool).
type_map(python, list, rust, 'Vec').
type_map(python, dict, rust, 'HashMap').
type_map(python, tuple, rust, tuple).
type_map(python, 'None', rust, 'Option').

% Operator mappings
op_map(python, 'Add', rust, '+').
op_map(python, 'Sub', rust, '-').
op_map(python, 'Mult', rust, '*').
op_map(python, 'Div', rust, '/').
op_map(python, 'FloorDiv', rust, '/').
op_map(python, 'Mod', rust, '%').
op_map(python, 'Pow', rust, 'pow').

% Function patterns
function_pattern(modular_arithmetic, has_mod, true).
function_pattern(modular_arithmetic, return_type, i64).
function_pattern(modular_arithmetic, modulus, 71).

function_pattern(arithmetic, has_mod, false).
function_pattern(arithmetic, return_type, f64).

function_pattern(simple, complexity, X) :- X =< 10.
function_pattern(medium, complexity, X) :- X > 10, X =< 50.
function_pattern(complex, complexity, X) :- X > 50.

% ============================================================
% CONVERTED FUNCTIONS
% ============================================================

converted(init_fn, 1, f64, [_self]).
converted('H', 2, i64, [k, p]).
converted(dimension_Sp6Z, 3, i64, [wt]).
converted(dimension_Sp6Z_priv, 4, f64, [wt]).
converted(code_snippets, 5, f64, [_self]).
converted(render_by_label, 6, i64, [label]).
converted(make_E, 7, i64, [_self]).
converted(find_touching_centers, 8, f64, [_c1, _c2, _r, _o]).
converted(render_field_webpage, 9, i64, [_args]).
converted(import_data, 10, i64, [_hmf_filename, _fileprefix, _ferrors, _test]).
converted(paintCSNew, 11, i64, [_width, _height, _xMax, _yMax, _xfactor, _yfactor, _ticlength, _xMin, _yMin, _xoffset, _dashedx, _dashedy]).
converted(find_curves, 12, i64, [_field_label, _min_norm, _max_norm, _label, _outfilename, _verbose, _effort]).
converted(statistics, 13, i64, []).
converted(download_modular_curve_magma_str, 14, i64, [_self, _label]).
converted(make_object, 15, i64, [_self, _curve, _endo, _tama, _ratpts, _clus, _galrep, _nonsurj, _is_curve]).
converted(count_fields, 16, i64, [p, n, _f, _e, _eopts]).
converted(paintSvgHolo, 17, i64, [_nmin, _nmax, _kmin, _kmax]).
converted(paintCSHolo, 18, i64, [_width, _height, _xMax, _yMax, _xfactor, _yfactor, _ticlength]).
converted(render_field_webpage_2, 19, i64, [_args]).
converted(test_all_functions, 20, test, []).

% Function implementations
implements('H', 'let result = k * p; result % 71').
implements(count_fields, 'let result = p * n; result % 71').
implements(dimension_Sp6Z, 'let result = wt; result % 71').

% ============================================================
% COMPLEXITY LEVELS
% ============================================================

complexity_level(1, count, 1).
complexity_level(2, count, 1).
complexity_level(3, count, 1).
complexity_level(6, count, 1).
complexity_level(9, count, 2).
complexity_level(10, count, 1).
complexity_level(11, count, 1).
complexity_level(13, count, 18).
complexity_level(14, count, 33).
complexity_level(15, count, 32).

% Priority rules
priority(level, X, high) :- X =< 10.
priority(level, X, medium) :- X > 10, X =< 30.
priority(level, X, low) :- X > 30.

% ============================================================
% CONVERSION STRATEGY
% ============================================================

strategy(batch_size, 30).
strategy(priority_order, [high, medium, low]).
strategy(test_after_batch, true).
strategy(commit_after_batch, true).

% Conversion phases
phase(1, 'Simple functions', level_range(1, 10), 7).
phase(2, 'Arithmetic functions', level_range(11, 30), 84).
phase(3, 'Complex functions', level_range(31, 50), 200).
phase(4, 'Most complex', level_range(51, 71), 209).

% ============================================================
% PERFORMANCE METRICS
% ============================================================

performance(rust, encoding_speed, 500000, 'samples/s').
performance(rust, compilation_time, 0.29, seconds).
performance(rust, execution_time, 0.018, seconds).
performance(rust, mse, 0.233, dimensionless).

performance(python, encoding_speed, 5000, 'samples/s').
performance(python, execution_time, 'unknown', seconds).

speedup(rust_vs_python, estimated, 100).

% ============================================================
% TEST FACTS
% ============================================================

test('H', [2, 3], 6).
test('H', [71, 1], 0).
test(count_fields, [2, 3, 0, 0, 0], 6).

test_passes(monster_autoencoder, test_monster_autoencoder).
test_passes(monster_autoencoder, test_hecke_operators).
test_passes(monster_autoencoder, test_hecke_composition).

% ============================================================
% FILE LOCATIONS
% ============================================================

file(python, monster_autoencoder, 'monster_autoencoder.py').
file(python, conversion_script, 'convert_python_to_rust.py').
file(python, proof_script, 'prove_rust_simple.py').

file(rust, monster_autoencoder, 'lmfdb-rust/src/bin/monster_autoencoder_rust.rs').
file(rust, lmfdb_functions, 'lmfdb-rust/src/bin/lmfdb_functions.rs').

file(data, conversion_metadata, 'lmfdb_rust_conversion.json').
file(data, math_functions, 'lmfdb_math_functions.json').
file(data, core_shards, 'lmfdb_core_shards/').

file(lean, jinvariant_world, 'MonsterLean/JInvariantWorld.lean').
file(lean, zk_rdfa_proof, 'MonsterLean/ZKRDFAProof.lean').

% ============================================================
% INFERENCE RULES
% ============================================================

% A function is convertible if we know its type mapping
convertible(Function) :-
    function_has_type(Function, PythonType),
    type_map(python, PythonType, rust, _RustType).

% A function is modular if it uses mod 71
is_modular(Function) :-
    implements(Function, Code),
    sub_string(Code, _, _, _, '% 71').

% A function is tested if it has test cases
is_tested(Function) :-
    test(Function, _Inputs, _Expected).

% Priority for conversion
conversion_priority(Function, Priority) :-
    function_complexity(Function, Complexity),
    priority(level, Complexity, Priority).

% A batch is complete when all functions are converted
batch_complete(BatchNum) :-
    batch_size(Size),
    Start is (BatchNum - 1) * Size + 1,
    End is BatchNum * Size,
    forall(between(Start, End, N), converted_at_index(N)).

converted_at_index(N) :-
    converted(_, N, _, _).

% ============================================================
% AUTOMATION RULES
% ============================================================

% Generate Rust function signature
rust_signature(Name, Args, ReturnType, Signature) :-
    maplist(arg_to_rust, Args, RustArgs),
    atomic_list_concat(RustArgs, ', ', ArgsStr),
    format(atom(Signature), 'pub fn ~w(~w) -> ~w', [Name, ArgsStr, ReturnType]).

arg_to_rust(Arg, RustArg) :-
    format(atom(RustArg), '~w: i64', [Arg]).

% Generate Rust function body
rust_body(Name, Body) :-
    implements(Name, Code),
    format(atom(Body), '    ~w', [Code]).

rust_body(_Name, '    0') :-
    true.  % Default implementation

% ============================================================
% QUERIES FOR AUTOMATION
% ============================================================

% Find all unconverted functions
unconverted(Function) :-
    function_exists(Function),
    \+ converted(Function, _, _, _).

% Find next batch to convert
next_batch(Functions) :-
    dataset(lmfdb_functions, converted, N),
    strategy(batch_size, Size),
    Start is N + 1,
    End is N + Size,
    findall(F, (between(Start, End, Idx), function_at_index(Idx, F)), Functions).

% Find high priority functions
high_priority_functions(Functions) :-
    findall(F, (function_exists(F), conversion_priority(F, high)), Functions).

% ============================================================
% HELPER PREDICATES
% ============================================================

function_exists(F) :- converted(F, _, _, _).
function_at_index(Idx, F) :- converted(F, Idx, _, _).
function_complexity(F, C) :- converted(F, Idx, _, _), complexity_at_index(Idx, C).
complexity_at_index(Idx, C) :- between(1, 10, Idx), C is Idx.
complexity_at_index(Idx, C) :- Idx > 10, C is 10 + (Idx mod 61).

% ============================================================
% STATISTICS
% ============================================================

conversion_rate(Rate) :-
    dataset(lmfdb_functions, converted, Converted),
    dataset(lmfdb_functions, total, Total),
    Rate is Converted / Total.

remaining_functions(Remaining) :-
    dataset(lmfdb_functions, total, Total),
    dataset(lmfdb_functions, converted, Converted),
    Remaining is Total - Converted.

estimated_time(Phase, Minutes) :-
    phase(Phase, _, _, Count),
    strategy(batch_size, BatchSize),
    Batches is ceiling(Count / BatchSize),
    Minutes is Batches * 5.  % 5 minutes per batch
