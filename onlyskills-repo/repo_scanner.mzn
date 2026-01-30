% Repo Scanner Duplicate Detection - MiniZinc
include "globals.mzn";

% Parameters
int: n_repos = 100;  % Number of repos scanned
int: n_shards = 71;  % Monster shards

% Repo names (hashed to integers)
array[1..n_repos] of int: repo_hashes;

% Decision variables
array[1..n_repos] of var 1..n_shards: shard_assignment;

% Constraint: Shard assignment based on hash
constraint forall(i in 1..n_repos) (
  shard_assignment[i] = (repo_hashes[i] mod 71) + 1
);

% Constraint: Detect duplicates
array[1..n_repos, 1..n_repos] of var bool: is_duplicate;
constraint forall(i, j in 1..n_repos where i < j) (
  is_duplicate[i,j] = (repo_hashes[i] = repo_hashes[j])
);

% Count duplicates
var int: duplicate_count = sum(i, j in 1..n_repos where i < j)(
  bool2int(is_duplicate[i,j])
);

% Novelty: minimize duplicates
solve minimize duplicate_count;

% Output
output [
  "Duplicate count: \(duplicate_count)\n",
  "Novel: \(duplicate_count = 0)\n",
  "Shard distribution:\n"
] ++
[
  "  Shard \(s): \(count(shard_assignment, s))\n"
  | s in 1..n_shards where count(shard_assignment, s) > 0
];

% Data (would be loaded from file)
repo_hashes = [
  42, 17, 59, 23, 11, 47, 31, 71, 2, 3,
  5, 7, 13, 19, 29, 41, 43, 53, 61, 67,
  % ... 80 more
  1, 4, 6, 8, 9, 10, 12, 14, 15, 16,
  18, 20, 21, 22, 24, 25, 26, 27, 28, 30,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 44,
  45, 46, 48, 49, 50, 51, 52, 54, 55, 56,
  57, 58, 60, 62, 63, 64, 65, 66, 68, 69,
  70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
  81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
  91, 92, 93, 94, 95, 96, 97, 98, 99, 100
];
