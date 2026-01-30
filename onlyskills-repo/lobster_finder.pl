% Lobster Finder - Prolog implementation
:- module(lobster_finder, [
    analyze_chain/2,
    find_lobster/1,
    ingest_repo/2
]).

% Top Solana tokens
solana_token(sol, 'Solana', 'https://github.com/solana-labs/solana').
solana_token(bonk, 'Bonk', 'https://github.com/bonk-inu/bonk').
solana_token(jup, 'Jupiter', 'https://github.com/jup-ag/jupiter-core').
solana_token(wif, 'dogwifhat', 'https://github.com/dogwifhat/dogwifhat').
solana_token(pyth, 'Pyth Network', 'https://github.com/pyth-network/pyth-client').

% Top Ethereum tokens
ethereum_token(eth, 'Ethereum', 'https://github.com/ethereum/go-ethereum').
ethereum_token(usdt, 'Tether', 'https://github.com/tether-to/tether').
ethereum_token(usdc, 'USD Coin', 'https://github.com/circlefoundation/stablecoin-evm').
ethereum_token(link, 'Chainlink', 'https://github.com/smartcontractkit/chainlink').
ethereum_token(uni, 'Uniswap', 'https://github.com/Uniswap/v3-core').

% Lobster score calculation
lobster_score(Symbol, Score) :-
    (solana_token(Symbol, _, URL) ; ethereum_token(Symbol, _, URL)),
    score_components(URL, Components),
    sum_list(Components, Score).

score_components(URL, [Ecosystem, Meme, DeFi]) :-
    (sub_string(URL, _, _, _, "solana") -> Ecosystem = 0.3 ; Ecosystem = 0.0),
    (sub_string(URL, _, _, _, "bonk") ; sub_string(URL, _, _, _, "wif") -> Meme = 0.4 ; Meme = 0.0),
    (sub_string(URL, _, _, _, "uniswap") ; sub_string(URL, _, _, _, "jupiter") -> DeFi = 0.2 ; DeFi = 0.0).

% Lobster detection
is_lobster(Symbol) :- lobster_score(Symbol, Score), Score > 0.5.
is_mega_lobster(Symbol) :- lobster_score(Symbol, Score), Score > 0.7.

% The Lobster (highest score)
the_lobster(Symbol) :-
    lobster_score(Symbol, Score),
    \+ (lobster_score(_, OtherScore), OtherScore > Score).

% Prioritize by score
prioritized_tokens(Sorted) :-
    findall(Score-Symbol, lobster_score(Symbol, Score), Pairs),
    sort(0, @>=, Pairs, Sorted).

% Analyze chain
analyze_chain(solana, Tokens) :-
    findall(Symbol-Score, (solana_token(Symbol, _, _), lobster_score(Symbol, Score)), Tokens).

analyze_chain(ethereum, Tokens) :-
    findall(Symbol-Score, (ethereum_token(Symbol, _, _), lobster_score(Symbol, Score)), Tokens).

% Ingest repo (sandboxed)
ingest_repo(Symbol, OutputDir) :-
    (solana_token(Symbol, Name, URL) ; ethereum_token(Symbol, Name, URL)),
    format(atom(RepoDir), '~w/~w', [OutputDir, Symbol]),
    format('🦞 Ingesting ~w (~w)~n', [Symbol, Name]),
    
    % Clone via unshare (kernel sandbox)
    format(atom(Cmd), 'unshare --user --net --mount --map-root-user git clone --depth 1 ~w ~w/repo 2>/dev/null', [URL, RepoDir]),
    shell(Cmd),
    
    % Extract metadata
    lobster_score(Symbol, Score),
    format(atom(MetaFile), '~w/metadata.json', [RepoDir]),
    open(MetaFile, write, Stream),
    format(Stream, '{~n  "symbol": "~w",~n  "name": "~w",~n  "repo_url": "~w",~n  "lobster_score": ~w~n}~n', 
           [Symbol, Name, URL, Score]),
    close(Stream).

% Find and report the lobster
find_lobster(Report) :-
    the_lobster(Symbol),
    (solana_token(Symbol, Name, URL) ; ethereum_token(Symbol, Name, URL)),
    lobster_score(Symbol, Score),
    Report = [
        symbol(Symbol),
        name(Name),
        url(URL),
        score(Score)
    ].
