% Topological Sort of First Bits - Prolog
:- module(topological_sort_bits, [
    build_dependency_graph/2,
    topological_sort/2,
    verify_sort/2
]).

% Monster primes
monster_prime(I, P) :- 
    Primes = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71],
    nth0(I, Primes, P).

% Get predictor bits for a bit
predictor_bits(BitIndex, MaxBits, Predictors) :-
    findall(Pred,
        (monster_prime(I, Prime),
         Shift is 1 << I,
         Pred is BitIndex xor (Prime * Shift),
         Pred < MaxBits),
        Predictors).

% Build dependency graph
build_dependency_graph(MaxBits, Graph) :-
    findall(Bit-Predictors,
        (between(0, MaxBits, Bit),
         predictor_bits(Bit, MaxBits, Predictors)),
        Graph).

% Calculate in-degrees
in_degrees(Graph, InDegrees) :-
    findall(Bit-0, member(Bit-_, Graph), InitDegrees),
    foldl(update_in_degree, Graph, InitDegrees, InDegrees).

update_in_degree(Bit-Predictors, DegreesIn, DegreesOut) :-
    foldl(increment_degree, Predictors, DegreesIn, DegreesOut).

increment_degree(Pred, DegreesIn, DegreesOut) :-
    (select(Pred-D, DegreesIn, Rest) ->
        D1 is D + 1,
        DegreesOut = [Pred-D1|Rest]
    ; DegreesOut = DegreesIn).

% Topological sort (Kahn's algorithm)
topological_sort(Graph, Sorted) :-
    in_degrees(Graph, InDegrees),
    findall(Bit, member(Bit-0, InDegrees), Queue),
    topo_sort_loop(Graph, InDegrees, Queue, [], Sorted).

topo_sort_loop(_, _, [], Acc, Sorted) :- 
    reverse(Acc, Sorted).

topo_sort_loop(Graph, InDegrees, [Bit|Queue], Acc, Sorted) :-
    member(Bit-Predictors, Graph),
    update_degrees(Predictors, InDegrees, NewDegrees, NewQueue),
    append(Queue, NewQueue, UpdatedQueue),
    topo_sort_loop(Graph, NewDegrees, UpdatedQueue, [Bit|Acc], Sorted).

update_degrees([], Degrees, Degrees, []).
update_degrees([Pred|Preds], DegreesIn, DegreesOut, NewQueue) :-
    (select(Pred-D, DegreesIn, Rest) ->
        D1 is D - 1,
        (D1 = 0 ->
            NewQueue = [Pred|RestQueue],
            Degrees1 = [Pred-D1|Rest]
        ; NewQueue = RestQueue,
          Degrees1 = [Pred-D1|Rest])
    ; Degrees1 = DegreesIn,
      NewQueue = RestQueue),
    update_degrees(Preds, Degrees1, DegreesOut, RestQueue).

% Verify topological sort
verify_sort(Graph, Sorted) :-
    % Create position map
    findall(Bit-Pos, 
        (nth0(Pos, Sorted, Bit)),
        PositionMap),
    
    % Check all dependencies
    forall(
        (member(Bit-Predictors, Graph),
         member(Pred, Predictors),
         member(Bit-BitPos, PositionMap),
         member(Pred-PredPos, PositionMap)),
        PredPos < BitPos  % Predictor must come before
    ).

% Query topological order
query_order(MaxBits, Order) :-
    build_dependency_graph(MaxBits, Graph),
    topological_sort(Graph, Order).
