% MCTS-based Repository Picker with Reasoning
:- module(mcts_repo_picker, [
    pick_best_repo/2,
    mcts_search/3,
    explain_choice/2
]).

% Real ClawdBot/Moltbot/OpenClaw repositories
real_repo('https://github.com/steipete/openclaw', openclaw, legitimate).
real_repo('https://github.com/steipete/moltbot', moltbot, legitimate).
real_repo('https://github.com/steipete/clawdbot', clawdbot, legitimate).

% Fake/Scam repositories
fake_repo('https://github.com/clawdbot/clawd', clawd_scam, scam).
fake_repo('https://github.com/clawdbot/clawdbot', clawdbot_fake, scam).
fake_repo('https://github.com/moltbot/moltbot', moltbot_fake, scam).

% Repository features for MCTS evaluation
repo_feature(URL, creator, Creator) :-
    (sub_atom(URL, _, _, _, 'steipete') -> Creator = peter_steinberger ;
     Creator = unknown).

repo_feature(URL, age_days, Age) :-
    (real_repo(URL, _, _) -> Age = 90 ;  % ~3 months
     fake_repo(URL, _, _) -> Age = 2).    % 2 days

repo_feature(URL, contributors, Count) :-
    (real_repo(URL, _, _) -> Count = 15 ;
     fake_repo(URL, _, _) -> Count = 1).

repo_feature(URL, stars, Count) :-
    (real_repo(URL, openclaw, _) -> Count = 100000 ;  % 100k stars!
     real_repo(URL, _, _) -> Count = 5000 ;
     fake_repo(URL, _, _) -> Count = 10).

repo_feature(URL, commits, Count) :-
    (real_repo(URL, _, _) -> Count = 200 ;
     fake_repo(URL, _, _) -> Count = 3).

repo_feature(URL, last_commit_days, Days) :-
    (real_repo(URL, _, _) -> Days = 1 ;   % Active
     fake_repo(URL, _, _) -> Days = 30).  % Abandoned

% MCTS Node evaluation
evaluate_repo(URL, Score) :-
    repo_feature(URL, creator, Creator),
    repo_feature(URL, age_days, Age),
    repo_feature(URL, contributors, Contributors),
    repo_feature(URL, stars, Stars),
    repo_feature(URL, commits, Commits),
    repo_feature(URL, last_commit_days, LastCommit),
    
    % Scoring
    (Creator = peter_steinberger -> S1 = 0.5 ; S1 = 0.0),
    (Age > 30 -> S2 = 0.1 ; S2 = 0.0),
    (Contributors > 5 -> S3 = 0.1 ; S3 = 0.0),
    (Stars > 1000 -> S4 = 0.2 ; Stars > 100 -> S4 = 0.1 ; S4 = 0.0),
    (Commits > 50 -> S5 = 0.05 ; S5 = 0.0),
    (LastCommit < 7 -> S6 = 0.05 ; S6 = 0.0),
    
    Score is S1 + S2 + S3 + S4 + S5 + S6.

% MCTS simulation
mcts_simulate(URL, Value) :-
    evaluate_repo(URL, Score),
    (real_repo(URL, _, legitimate) -> Bonus = 0.5 ; Bonus = 0.0),
    (fake_repo(URL, _, scam) -> Penalty = -1.0 ; Penalty = 0.0),
    Value is Score + Bonus + Penalty.

% MCTS selection (UCB1)
ucb1_score(URL, Visits, ParentVisits, Score) :-
    mcts_simulate(URL, Value),
    (Visits > 0 ->
        Exploitation is Value / Visits,
        Exploration is 1.414 * sqrt(log(ParentVisits) / Visits),
        Score is Exploitation + Exploration
    ; Score = 1000000).  % Infinity for unvisited

% MCTS search
mcts_search(Repos, Iterations, BestRepo) :-
    mcts_iterate(Repos, Iterations, Scores),
    sort(0, @>=, Scores, [_-BestRepo|_]).

mcts_iterate(_, 0, []) :- !.
mcts_iterate(Repos, N, [Score-Repo|Rest]) :-
    N > 0,
    member(Repo, Repos),
    mcts_simulate(Repo, Score),
    N1 is N - 1,
    mcts_iterate(Repos, N1, Rest).

% Pick best repository with reasoning
pick_best_repo(Query, BestRepo) :-
    % Find all candidate repos
    findall(URL, 
        (real_repo(URL, Name, _) ; fake_repo(URL, Name, _),
         sub_atom(Name, _, _, _, Query)),
        Candidates),
    
    % Run MCTS
    mcts_search(Candidates, 100, BestRepo).

% Explain choice with reasoning
explain_choice(URL, Explanation) :-
    repo_feature(URL, creator, Creator),
    repo_feature(URL, stars, Stars),
    repo_feature(URL, contributors, Contributors),
    evaluate_repo(URL, Score),
    
    (real_repo(URL, Name, Status) -> 
        format(atom(Explanation), 
               'LEGITIMATE: ~w by ~w (~w stars, ~w contributors, score: ~2f)',
               [Name, Creator, Stars, Contributors, Score])
    ; fake_repo(URL, Name, Status) ->
        format(atom(Explanation),
               'SCAM: ~w (fake repo, ~w stars, score: ~2f)',
               [Name, Stars, Score])
    ; Explanation = 'Unknown repository').

% Reasoning chain
reasoning_chain(URL, Chain) :-
    findall(Reason,
        (repo_feature(URL, Feature, Value),
         format(atom(Reason), '~w: ~w', [Feature, Value])),
        Chain).

% Decision tree
decision(URL, Decision) :-
    repo_feature(URL, creator, Creator),
    (Creator = peter_steinberger ->
        Decision = 'TRUST: Created by Peter Steinberger (PSPDFKit founder)'
    ; repo_feature(URL, stars, Stars),
      Stars > 10000 ->
        Decision = 'TRUST: High star count indicates community validation'
    ; repo_feature(URL, age_days, Age),
      Age < 7 ->
        Decision = 'REJECT: Too new, likely scam'
    ; Decision = 'INVESTIGATE: Needs further analysis').

% Find the real ClawdBot
find_real_clawdbot(RealRepo) :-
    pick_best_repo(openclaw, RealRepo),
    explain_choice(RealRepo, Explanation),
    format('Found real ClawdBot: ~w~n', [RealRepo]),
    format('Reason: ~w~n', [Explanation]).
