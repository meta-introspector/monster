% Extract git repos from onlyskills donations
:- module(onlyskills_git_finder, [
    extract_repos/1,
    find_crypto_repos/1,
    assess_repo_risk/2
]).

:- use_module(library(http/json)).

% Load skill donations
load_donations(Donations) :-
    open('skill_donations.json', read, Stream),
    json_read(Stream, JSON),
    close(Stream),
    member(sample_donations=Donations, JSON).

% Extract git commits from donations
extract_git_commits(Commits) :-
    load_donations(Donations),
    findall(Commit-Donor-Skill,
        (member(Donation, Donations),
         member(git_commit=Commit, Donation),
         member(donor=Donor, Donation),
         member(skill_id=Skill, Donation)),
        Commits).

% Infer repo URLs from donors
donor_repo(pipelight_dev, 'https://github.com/pipelight/pipelight').
donor_repo(solana_labs, 'https://github.com/solana-labs/solana').
donor_repo(ethereum_foundation, 'https://github.com/ethereum/go-ethereum').
donor_repo(uniswap, 'https://github.com/Uniswap/v3-core').

% Extract all repos from onlyskills
extract_repos(Repos) :-
    findall(Donor-URL,
        (extract_git_commits(Commits),
         member(_-Donor-_, Commits),
         donor_repo(Donor, URL)),
        RepoList),
    sort(RepoList, Repos).

% Find crypto-related repos
find_crypto_repos(CryptoRepos) :-
    extract_repos(Repos),
    findall(Donor-URL,
        (member(Donor-URL, Repos),
         (sub_atom(URL, _, _, _, solana) ;
          sub_atom(URL, _, _, _, ethereum) ;
          sub_atom(URL, _, _, _, uniswap) ;
          sub_atom(URL, _, _, _, bonk))),
        CryptoRepos).

% Assess repo risk using MCTS
assess_repo_risk(URL, Risk) :-
    % Extract repo characteristics
    (sub_atom(URL, _, _, _, solana) -> ChainRisk = 0.2 ; ChainRisk = 0.1),
    (sub_atom(URL, _, _, _, bonk) -> MemeRisk = 0.3 ; MemeRisk = 0.0),
    (sub_atom(URL, _, _, _, uniswap) -> DeFiRisk = 0.1 ; DeFiRisk = 0.0),
    
    % Aggregate risk
    Risk is ChainRisk + MemeRisk + DeFiRisk.

% Create prediction market from onlyskills data
create_market_from_onlyskills(Donor, Market) :-
    donor_repo(Donor, URL),
    assess_repo_risk(URL, Risk),
    (Risk > 0.6 -> Zone = 59 ;
     Risk > 0.4 -> Zone = 47 ;
     Risk > 0.2 -> Zone = 31 ;
     Zone = 11),
    Market = [
        donor(Donor),
        repo_url(URL),
        risk(Risk),
        zone(Zone)
    ].

% Find the lobster in onlyskills donations
find_onlyskills_lobster(Lobster) :-
    findall(Risk-Market,
        (donor_repo(Donor, _),
         create_market_from_onlyskills(Donor, Market),
         member(risk(Risk), Market)),
        Markets),
    sort(0, @<, Markets, [_-Lobster|_]).
