% Top Solana Meme Coins Model
:- module(solana_meme_coins, [
    top_coin/4,
    assess_lobster_potential/2,
    find_the_lobster/1
]).

% Top Solana meme coins (Symbol, Name, MarketCap, RepoURL)
top_coin(bonk, 'Bonk', 380000000, 'https://github.com/bonk-inu/bonk').
top_coin(wif, 'dogwifhat', 380000000, 'https://github.com/dogwifhat/dogwifhat').
top_coin(popcat, 'Popcat', 150000000, 'https://github.com/popcat-meme/popcat').
top_coin(pengu, 'Pudgy Penguins', 120000000, 'https://github.com/pudgypenguins/pengu').
top_coin(clawd, 'Clawdbot', 8650000, 'https://github.com/clawdbot/clawd').

% CLAWD is a SCAM - unauthorized token
scam_token(clawd, 'Unauthorized meme coin, creator denounced it').

% Lobster potential factors
lobster_factor(bonk, community, 0.9).  % Strong community
lobster_factor(bonk, liquidity, 0.8).  % High liquidity
lobster_factor(bonk, meme_power, 0.9). % OG Solana meme

lobster_factor(wif, community, 0.9).   % Viral sensation
lobster_factor(wif, liquidity, 0.8).   % High volume
lobster_factor(wif, meme_power, 1.0).  % Dog with hat = peak meme

lobster_factor(popcat, community, 0.7).
lobster_factor(popcat, liquidity, 0.6).
lobster_factor(popcat, meme_power, 0.8).

lobster_factor(pengu, community, 0.8). % NFT-native
lobster_factor(pengu, liquidity, 0.7).
lobster_factor(pengu, meme_power, 0.7).

lobster_factor(clawd, community, 0.1). % SCAM
lobster_factor(clawd, liquidity, 0.2).
lobster_factor(clawd, meme_power, 0.0).

% Assess lobster potential
assess_lobster_potential(Coin, Score) :-
    findall(Factor, lobster_factor(Coin, _, Factor), Factors),
    sum_list(Factors, Sum),
    length(Factors, Count),
    (Count > 0 -> Score is Sum / Count ; Score = 0.0).

% Risk assessment
risk_score(Coin, Risk) :-
    (scam_token(Coin, _) -> Risk = 1.0 ;  % Max risk for scams
     top_coin(Coin, _, MarketCap, _),
     (MarketCap > 300000000 -> Risk = 0.2 ;  % Low risk, established
      MarketCap > 100000000 -> Risk = 0.4 ;  % Medium risk
      Risk = 0.7)).                           % High risk, small cap

% Zone assignment
assign_zone(Coin, Zone) :-
    risk_score(Coin, Risk),
    (Risk > 0.8 -> Zone = 71 ;  % Catastrophic (scams)
     Risk > 0.6 -> Zone = 59 ;  % Critical
     Risk > 0.4 -> Zone = 47 ;  % High
     Risk > 0.2 -> Zone = 31 ;  % Medium
     Zone = 11).                 % Low

% Find THE LOBSTER
find_the_lobster(Lobster) :-
    findall(Score-Coin,
        (top_coin(Coin, _, _, _),
         \+ scam_token(Coin, _),  % Exclude scams
         assess_lobster_potential(Coin, Score)),
        Scores),
    sort(0, @>=, Scores, [_-Lobster|_]).

% Market analysis
market_analysis(Coin, Analysis) :-
    top_coin(Coin, Name, MarketCap, URL),
    assess_lobster_potential(Coin, LobsterScore),
    risk_score(Coin, Risk),
    assign_zone(Coin, Zone),
    (scam_token(Coin, Reason) -> Scam = Reason ; Scam = 'legitimate'),
    Analysis = [
        symbol(Coin),
        name(Name),
        market_cap(MarketCap),
        repo_url(URL),
        lobster_score(LobsterScore),
        risk(Risk),
        zone(Zone),
        scam_status(Scam)
    ].
