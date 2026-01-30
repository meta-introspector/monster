% Semantic 71-Enum with Context Bit Prediction - Prolog
:- module(semantic_71, [
    predict_first_bit/3,
    semantic_to_shard/2,
    first_bit_depth/2,
    query_by_depth/2
]).

% 71 semantic categories (Monster primes)
semantic_category(71, proof).
semantic_category(59, theorem).
semantic_category(47, verified).
semantic_category(41, correct).
semantic_category(31, optimal).
semantic_category(29, efficient).
semantic_category(23, elegant).
semantic_category(19, simple).
semantic_category(17, clear).
semantic_category(13, useful).
semantic_category(11, working).
semantic_category(7, good).
semantic_category(5, basic).
semantic_category(3, minimal).
semantic_category(2, raw).

% Predict first bit from context bits
predict_first_bit(ContextBits, Semantic, FirstBit) :-
    % XOR context with semantic
    Prediction is (ContextBits xor Semantic) /\ 1,
    FirstBit = Prediction.

% Map semantic to shard (bucket)
semantic_to_shard(Semantic, Shard) :-
    Shard is Semantic mod 71.

% Calculate first bit depth (leading zeros)
first_bit_depth(ContextBits, Depth) :-
    % Count leading zeros
    (ContextBits = 0 ->
        Depth = 64
    ; msb_position(ContextBits, MSB),
      Depth is 63 - MSB
    ).

% Find MSB position
msb_position(N, Pos) :-
    N > 0,
    msb_position(N, 0, Pos).

msb_position(0, Pos, Pos) :- !.
msb_position(N, Acc, Pos) :-
    N > 0,
    N1 is N >> 1,
    Acc1 is Acc + 1,
    msb_position(N1, Acc1, Pos).

% Query entries by depth
query_by_depth(Depth, Entries) :-
    findall([semantic(S), depth(D), shard(Shard)],
        (semantic_category(S, _),
         atom_codes(S, Codes),
         sum_list(Codes, ContextBits),
         first_bit_depth(ContextBits, D),
         D = Depth,
         semantic_to_shard(S, Shard)),
        Entries).

% Generate context bits from name
name_to_context(Name, ContextBits) :-
    atom_codes(Name, Codes),
    sum_list(Codes, ContextBits).

% Full semantic entry
semantic_entry(Name, Entry) :-
    semantic_category(Semantic, Name),
    name_to_context(Name, ContextBits),
    predict_first_bit(ContextBits, Semantic, FirstBit),
    first_bit_depth(ContextBits, Depth),
    semantic_to_shard(Semantic, Shard),
    Entry = [
        semantic(Semantic),
        name(Name),
        context_bits(ContextBits),
        first_bit(FirstBit),
        depth(Depth),
        shard(Shard)
    ].

% Statistics
semantic_stats(Stats) :-
    findall(Entry, semantic_entry(_, Entry), Entries),
    length(Entries, Total),
    findall(D, (member(E, Entries), member(depth(D), E)), Depths),
    sum_list(Depths, SumDepth),
    AvgDepth is SumDepth / Total,
    Stats = [
        total(Total),
        avg_depth(AvgDepth)
    ].
