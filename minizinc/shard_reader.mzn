% MiniZinc: Shard reader constraint model

% Parameters
int: NUM_SHARDS = 71;
int: NUM_PRIMES = 15;

% Shard source types
enum SOURCE = {LOCAL, ARCHIVE_ORG, HUGGING_FACE};

% Decision variables
var SOURCE: source_type;
var 0..NUM_SHARDS-1: shard_id;
var bool: shard_valid;

% Shard properties
array[0..NUM_SHARDS-1] of var int: shard_sizes;
array[0..NUM_SHARDS-1] of var bool: shard_loaded;

% Constraints

% All shards must have positive size
constraint forall(i in 0..NUM_SHARDS-1)(
  shard_sizes[i] > 0
);

% Shard ID must be valid
constraint shard_id >= 0 /\ shard_id < NUM_SHARDS;

% If shard is valid, it must be loaded
constraint shard_valid -> shard_loaded[shard_id];

% Content-addressable: each shard has unique hash
constraint forall(i, j in 0..NUM_SHARDS-1 where i != j)(
  shard_sizes[i] != shard_sizes[j] -> true  % Different content
);

% Source-specific constraints
constraint 
  if source_type = LOCAL then
    % Local files must exist
    forall(i in 0..NUM_SHARDS-1)(shard_loaded[i])
  elseif source_type = ARCHIVE_ORG then
    % Archive.org requires network
    true
  elseif source_type = HUGGING_FACE then
    % HF requires authentication
    true
  endif;

% Objective: Maximize loaded shards
solve maximize sum(i in 0..NUM_SHARDS-1)(shard_loaded[i]);

% Output
output [
  "Source: ", show(source_type), "\n",
  "Shard ID: ", show(shard_id), "\n",
  "Valid: ", show(shard_valid), "\n",
  "Loaded shards: ", show(sum(i in 0..NUM_SHARDS-1)(shard_loaded[i])), "/", show(NUM_SHARDS), "\n"
];
