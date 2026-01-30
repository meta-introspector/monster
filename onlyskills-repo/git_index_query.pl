% Query Git Repos from Shared Memory
:- module(git_index_query, [
    query_git_repos/2,
    repos_by_shard/2,
    count_repos/1
]).

% Shared memory path
git_index_path('/dev/shm/monster_git_index').

% Query repos by shard
repos_by_shard(Shard, Repos) :-
    git_index_path(Path),
    % Would read from shared memory via FFI
    % For now, simulate
    findall(Repo,
        (between(0, 100, I),
         Hash is I * 1000,
         RepoShard is Hash mod 71,
         RepoShard = Shard,
         format(atom(Repo), 'repo_~w', [I])),
        Repos).

% Count total repos
count_repos(Count) :-
    % Would read from shared memory header
    Count = 609.  % From earlier scan

% Query by pattern
query_git_repos(Pattern, Repos) :-
    % Would search shared memory
    findall(Repo,
        (repos_by_shard(_, AllRepos),
         member(Repo, AllRepos),
         sub_atom(Repo, _, _, _, Pattern)),
        Repos).

% Find repos in specific shard range
repos_in_range(MinShard, MaxShard, Repos) :-
    findall(Repo,
        (between(MinShard, MaxShard, Shard),
         repos_by_shard(Shard, ShardRepos),
         member(Repo, ShardRepos)),
        Repos).

% Statistics
repo_stats(Stats) :-
    count_repos(Total),
    findall(Shard-Count,
        (between(0, 70, Shard),
         repos_by_shard(Shard, Repos),
         length(Repos, Count),
         Count > 0),
        Distribution),
    Stats = [
        total(Total),
        distribution(Distribution)
    ].
