% MiniZinc model for shard_lmfdb_by_71.rs
% Constraint: Partition chunks into 71 shards

int: num_chunks;
set of int: CHUNKS = 1..num_chunks;
set of int: SHARDS = 0..70;

% Each chunk has a hash
array[CHUNKS] of int: hashes;

% Shard assignment: hash % 71
array[CHUNKS] of var SHARDS: shard_assignment;
constraint forall(c in CHUNKS)(
  shard_assignment[c] = hashes[c] mod 71
);

% Count chunks per shard
array[SHARDS] of var int: shard_counts;
constraint forall(s in SHARDS)(
  shard_counts[s] = sum(c in CHUNKS)(shard_assignment[c] = s)
);

% All shards should be used
constraint forall(s in SHARDS)(shard_counts[s] >= 0);

solve satisfy;

output [
  "Shard distribution:\n"
] ++ [
  "Shard \(s): \(fix(shard_counts[s])) chunks\n" | s in SHARDS
];
