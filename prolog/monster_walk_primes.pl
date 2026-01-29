% Prolog: Monster Walk with Prime Factorizations in All Bases

% Monster prime factorization
monster_primes([
    (2, 46), (3, 20), (5, 9), (7, 6), (11, 2), (13, 3),
    (17, 1), (19, 1), (23, 1), (29, 1), (31, 1),
    (41, 1), (47, 1), (59, 1), (71, 1)
]).

% Step 2: Remove 17, 59
step2_removed([(17, 1), (59, 1)]).

% Step 4: Remove 8 factors (Group 1)
step4_removed([
    (7, 6), (11, 2), (17, 1), (19, 1),
    (29, 1), (31, 1), (41, 1), (59, 1)
]).

% Step 6: Remove 4 factors (Group 2)
step6_removed([(3, 20), (5, 9), (13, 3), (31, 1)]).

% Step 8: Remove 4 factors (Group 3)
step8_removed([(3, 20), (13, 3), (31, 1), (71, 1)]).

% Remove primes from factorization
remove_primes([], Remaining, Remaining).
remove_primes([(P, _)|Rest], Current, Result) :-
    select((P, _), Current, NewCurrent),
    remove_primes(Rest, NewCurrent, Result).

% Step 4 remaining primes
step4_remaining(Remaining) :-
    monster_primes(Monster),
    step4_removed(Removed),
    remove_primes(Removed, Monster, Remaining).

% Convert to base
to_base(0, _, [0]).
to_base(N, Base, [Digit|RestDigits]) :-
    N > 0,
    Digit is N mod Base,
    N1 is N // Base,
    to_base(N1, Base, RestDigits).

% Walk step
walk_step(StepNum, Removed, Remaining, Base, Representation) :-
    % Compute value from remaining primes
    % Convert to base
    to_base(Value, Base, Representation).

% Complete walk in base
walk_in_base(Base, [
    step(1, [], Monster),
    step(2, Removed2, Remaining2),
    step(4, Removed4, Remaining4),
    step(6, Removed6, Remaining6),
    step(8, Removed8, Remaining8),
    step(10, Monster, [(71, 1)])
]) :-
    monster_primes(Monster),
    step2_removed(Removed2),
    step4_removed(Removed4),
    step6_removed(Removed6),
    step8_removed(Removed8),
    remove_primes(Removed2, Monster, Remaining2),
    remove_primes(Removed4, Monster, Remaining4),
    remove_primes(Removed6, Monster, Remaining6),
    remove_primes(Removed8, Monster, Remaining8).

% Generate all bases
walk_all_bases(AllWalks) :-
    findall((Base, Walk),
        (between(2, 71, Base), walk_in_base(Base, Walk)),
        AllWalks).

% Query: Get walk in base 10
?- walk_in_base(10, Walk).

% Query: Get all walks
?- walk_all_bases(Walks), length(Walks, N).
% N = 70

% Query: Get step 4 remaining primes
?- step4_remaining(Remaining).
% Remaining = [(2,46), (3,20), (5,9), (13,3), (23,1), (47,1), (71,1)]
