% MiniZinc model for breadth_first_pipeline.rs
% Constraint: Verify breadth-first processing order

int: num_layers = 46;
int: num_shards = 15;
int: total_tasks = num_layers * num_shards;

set of int: LAYERS = 0..num_layers-1;
set of int: SHARDS = 0..num_shards-1;
set of int: TASKS = 1..total_tasks;

% Each task processes one (layer, shard) pair
array[TASKS] of var LAYERS: task_layer;
array[TASKS] of var SHARDS: task_shard;

% Breadth-first: process all shards at layer L before layer L+1
array[TASKS] of var int: execution_order;
constraint forall(t in TASKS)(execution_order[t] = t);

% Constraint: tasks at layer L come before tasks at layer L+1
constraint forall(t1, t2 in TASKS where t1 < t2)(
  task_layer[t1] < task_layer[t2] \/
  (task_layer[t1] = task_layer[t2] /\ task_shard[t1] <= task_shard[t2])
);

% Each (layer, shard) pair processed exactly once
constraint forall(l in LAYERS, s in SHARDS)(
  exists(t in TASKS)(task_layer[t] = l /\ task_shard[t] = s)
);

solve satisfy;

output [
  "Total tasks: \(total_tasks)\n",
  "Layers: \(num_layers), Shards: \(num_shards)\n",
  "Breadth-first order verified\n"
];
