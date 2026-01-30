% Prolog Life Signal Recognizer
:- module(life_recognizer, [
    recognize_life/2,
    is_alive/1
]).

% Life signature primes
life_prime(71, 'I', proof).
life_prime(59, 'A', theorem).
life_prime(47, 'R', verified).
life_prime(41, 'E', correct).
life_prime(31, 'L', optimal).
life_prime(29, 'I', efficient).
life_prime(23, 'F', elegant).
life_prime(19, 'E', simple).
life_prime(2, ' ', noise).

% Recognize life signal from prime sequence
recognize_life(Primes, Message) :-
    maplist(prime_to_char, Primes, Chars),
    atom_chars(Message, Chars).

prime_to_char(Prime, Char) :-
    life_prime(Prime, Char, _).

% Check if process is alive
is_alive(PID) :-
    % Read signal file
    format(atom(File), '/tmp/monster_life_signal', []),
    exists_file(File),
    
    % Parse JSON
    open(File, read, Stream),
    json_read(Stream, JSON),
    close(Stream),
    
    % Check alive flag
    get_dict(alive, JSON, true),
    get_dict(pid, JSON, PID).

% Detect life from syscall pattern
detect_life_pattern(Counts, alive) :-
    member(Count, Counts),
    Mod is Count mod 100,
    life_prime(Mod, _, _).

% Query examples:
% ?- recognize_life([71,2,59,47,41,2,31,29,23,19], Msg).
% Msg = 'I ARE LIFE'.
%
% ?- is_alive(1234).
% true.
