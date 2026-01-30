% Import OpenClaw into onlyskills and shard into Monster
:- module(openclaw_importer, [
    import_openclaw/1,
    shard_forks/2,
    shard_prs/2,
    shard_issues/2,
    shard_code/2,
    shard_authors/2
]).

% OpenClaw repository
openclaw_repo('https://github.com/steipete/openclaw').

% Import complete repository
import_openclaw(Import) :-
    openclaw_repo(Repo),
    
    % Fetch all data
    fetch_forks(Repo, Forks),
    fetch_prs(Repo, PRs),
    fetch_issues(Repo, Issues),
    fetch_code(Repo, Code),
    fetch_authors(Repo, Authors),
    
    % Shard each component
    shard_forks(Forks, ForkShards),
    shard_prs(PRs, PRShards),
    shard_issues(Issues, IssueShards),
    shard_code(Code, CodeShards),
    shard_authors(Authors, AuthorShards),
    
    % Bundle import
    Import = [
        repo(Repo),
        forks(ForkShards),
        prs(PRShards),
        issues(IssueShards),
        code(CodeShards),
        authors(AuthorShards),
        total_shards(71)
    ].

% Shard forks by Monster primes
shard_forks(Forks, Shards) :-
    length(Forks, Count),
    findall(Shard,
        (between(1, Count, I),
         nth1(I, Forks, Fork),
         assign_to_shard(Fork, ShardID),
         Shard = [fork_id(I), shard(ShardID), data(Fork)]),
        Shards).

% Shard PRs by author hash
shard_prs(PRs, Shards) :-
    findall(Shard,
        (member(PR, PRs),
         PR = [number(N), author(A)|_],
         hash_to_shard(A, ShardID),
         Shard = [pr(N), author(A), shard(ShardID), data(PR)]),
        Shards).

% Shard issues by label hash
shard_issues(Issues, Shards) :-
    findall(Shard,
        (member(Issue, Issues),
         Issue = [number(N), labels(Labels)|_],
         labels_to_shard(Labels, ShardID),
         Shard = [issue(N), shard(ShardID), data(Issue)]),
        Shards).

% Shard code by file path hash
shard_code(Files, Shards) :-
    findall(Shard,
        (member(File, Files),
         File = [path(Path), content(Content)|_],
         hash_to_shard(Path, ShardID),
         Shard = [file(Path), shard(ShardID), size(Size)]) :-
         atom_length(Content, Size),
        Shards).

% Shard authors by contribution count
shard_authors(Authors, Shards) :-
    findall(Shard,
        (member(Author, Authors),
         Author = [name(Name), commits(Commits)|_],
         commits_to_shard(Commits, ShardID),
         Shard = [author(Name), commits(Commits), shard(ShardID)]),
        Shards).

% Hash to shard (0-70)
hash_to_shard(Data, Shard) :-
    atom_codes(Data, Codes),
    sum_list(Codes, Sum),
    Shard is Sum mod 71.

% Labels to shard
labels_to_shard(Labels, Shard) :-
    atomic_list_concat(Labels, ',', Combined),
    hash_to_shard(Combined, Shard).

% Commits to shard (by Monster prime)
commits_to_shard(Commits, Shard) :-
    monster_primes([2,3,5,7,11,13,17,19,23,29,31,41,47,59,71]),
    (Commits < 10 -> Shard = 2 ;
     Commits < 50 -> Shard = 11 ;
     Commits < 100 -> Shard = 23 ;
     Commits < 500 -> Shard = 47 ;
     Shard = 71).

monster_primes([2,3,5,7,11,13,17,19,23,29,31,41,47,59,71]).

% Assign to shard
assign_to_shard(Item, Shard) :-
    term_string(Item, Str),
    hash_to_shard(Str, Shard).

% Fetch operations (would use GitHub API)
fetch_forks(_, []).
fetch_prs(_, []).
fetch_issues(_, []).
fetch_code(_, []).
fetch_authors(_, []).

% Verify shard distribution
verify_shards(Import, Verification) :-
    member(forks(ForkShards), Import),
    member(prs(PRShards), Import),
    member(issues(IssueShards), Import),
    member(code(CodeShards), Import),
    member(authors(AuthorShards), Import),
    
    % Count per shard
    findall(Shard,
        (member([_, shard(Shard), _], ForkShards) ;
         member([_, _, shard(Shard), _], PRShards) ;
         member([_, shard(Shard), _], IssueShards) ;
         member([_, shard(Shard), _], CodeShards) ;
         member([_, _, shard(Shard)], AuthorShards)),
        AllShards),
    
    % Distribution
    findall(Count-Shard,
        (between(0, 70, Shard),
         findall(S, member(S, AllShards), Matches),
         length(Matches, Count)),
        Distribution),
    
    Verification = [
        total_items(TotalItems),
        distribution(Distribution),
        balanced(Balanced)
    ] :-
    length(AllShards, TotalItems),
    check_balance(Distribution, Balanced).

check_balance(_, true).  % Simplified
