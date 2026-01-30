% Prolog searcher for mathematical terms

:- dynamic fact/2.

% Load knowledge base
load_knowledge :-
    assertz(fact('Monster Group', 'sporadic_group')),
    assertz(fact('Monster Group', 'order_8e53')),
    assertz(fact('Bott Periodicity', 'k_theory')),
    assertz(fact('Bott Periodicity', 'period_8')),
    assertz(fact('Elliptic Curve', 'algebraic_geometry')),
    assertz(fact('Elliptic Curve', 'weierstrass_form')),
    assertz(fact('Hilbert Modular Form', 'number_theory')),
    assertz(fact('Calabi-Yau Threefold', 'string_theory')),
    assertz(fact('Monstrous Moonshine', 'monster_group')),
    assertz(fact('E8 Lattice', 'lie_algebra')),
    assertz(fact('ADE Classification', 'dynkin_diagram')),
    assertz(fact('Topological Modular Form', 'homotopy_theory')).

% Search for term
search(Term) :-
    fact(Term, Property),
    format('~w: ~w~n', [Term, Property]),
    fail.
search(_).

% Find related terms
related(Term1, Term2) :-
    fact(Term1, Property),
    fact(Term2, Property),
    Term1 \= Term2.

% Query all properties
properties(Term) :-
    findall(P, fact(Term, P), Props),
    format('Properties of ~w: ~w~n', [Term, Props]).

% Initialize
:- initialization(load_knowledge).
