% zkprologml: MCTS Risk Prediction Market in Prolog
:- module(mcts_risk_market, [
    mcts_predict_risk/3,
    create_market/4,
    find_lobster/1
]).

% MCTS node structure
:- dynamic mcts_node/5.  % mcts_node(NodeId, State, Visits, Value, Parent)
:- dynamic mcts_child/2. % mcts_child(ParentId, ChildId)

% Risk factors
risk_factor(unsafe_code, 0.3).
risk_factor(network_calls, 0.2).
risk_factor(process_spawn, 0.2).
risk_factor(few_commits, 0.2).
risk_factor(few_authors, 0.1).

% UCB1 calculation
ucb1(NodeId, ParentVisits, Score) :-
    mcts_node(NodeId, _, Visits, Value, _),
    (Visits = 0 ->
        Score = 1000000 ;  % Infinity
        Exploitation is Value / Visits,
        ExplorationTerm is 1.414 * sqrt(log(ParentVisits) / Visits),
        Score is Exploitation + ExplorationTerm
    ).

% Select best child using UCB1
select_best_child(ParentId, BestChild) :-
    mcts_node(ParentId, _, ParentVisits, _, _),
    findall(Score-ChildId, 
        (mcts_child(ParentId, ChildId), ucb1(ChildId, ParentVisits, Score)),
        Scores),
    sort(0, @>=, Scores, [_-BestChild|_]).

% Simulate risk outcome
simulate_risk(State, Risk) :-
    findall(R, 
        (risk_factor(Factor, R), sub_atom(State, _, _, _, Factor)),
        Risks),
    sum_list(Risks, TotalRisk),
    Risk is min(TotalRisk, 1.0).

% Backpropagate value
backpropagate(NodeId, Value) :-
    mcts_node(NodeId, State, Visits, OldValue, Parent),
    NewVisits is Visits + 1,
    NewValue is OldValue + Value,
    retract(mcts_node(NodeId, State, Visits, OldValue, Parent)),
    assertz(mcts_node(NodeId, State, NewVisits, NewValue, Parent)),
    (Parent \= none -> backpropagate(Parent, Value) ; true).

% Run MCTS iterations
mcts_iterate(0, _) :- !.
mcts_iterate(N, RootId) :-
    N > 0,
    select_best_child(RootId, Selected),
    simulate_risk(Selected, Risk),
    backpropagate(Selected, Risk),
    N1 is N - 1,
    mcts_iterate(N1, RootId).

% Predict risk using MCTS
mcts_predict_risk(Asset, Chain, RiskScore) :-
    % Initialize root
    format(atom(InitialState), '~w_~w', [Chain, Asset]),
    assertz(mcts_node(0, InitialState, 0, 0.0, none)),
    
    % Create children
    assertz(mcts_node(1, 'safe_path', 0, 0.0, 0)),
    assertz(mcts_node(2, 'risky_path', 0, 0.0, 0)),
    assertz(mcts_child(0, 1)),
    assertz(mcts_child(0, 2)),
    
    % Run iterations
    mcts_iterate(1000, 0),
    
    % Get best score
    mcts_node(0, _, Visits, Value, _),
    (Visits > 0 -> RiskScore is Value / Visits ; RiskScore = 0.5),
    
    % Cleanup
    retractall(mcts_node(_, _, _, _, _)),
    retractall(mcts_child(_, _)).

% Map risk to threat level and zone
threat_level(Risk, catastrophic, 71) :- Risk > 0.8, !.
threat_level(Risk, critical, 59) :- Risk > 0.6, !.
threat_level(Risk, high, 47) :- Risk > 0.4, !.
threat_level(Risk, medium, 31) :- Risk > 0.2, !.
threat_level(_, low, 11).

% Create prediction market
create_market(Asset, Chain, RepoURL, Market) :-
    mcts_predict_risk(Asset, Chain, Risk),
    threat_level(Risk, Level, Zone),
    Confidence is 1.0 - abs(Risk - 0.5) * 2.0,
    Market = [
        asset(Asset),
        chain(Chain),
        repo_url(RepoURL),
        risk_probability(Risk),
        confidence(Confidence),
        threat_level(Level),
        zone(Zone)
    ].

% Find the lobster (lowest risk)
find_lobster(Lobster) :-
    findall(Risk-Market,
        (coin(Asset, Chain, URL), 
         create_market(Asset, Chain, URL, Market),
         member(risk_probability(Risk), Market)),
        Markets),
    sort(0, @<, Markets, [_-Lobster|_]).

% Coin database
coin(sol, solana, 'https://github.com/solana-labs/solana').
coin(bonk, solana, 'https://github.com/bonk-inu/bonk').
coin(eth, ethereum, 'https://github.com/ethereum/go-ethereum').
coin(uni, ethereum, 'https://github.com/Uniswap/v3-core').
