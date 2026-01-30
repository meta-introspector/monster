% MiniZinc model for extract_71_objects.rs
% Constraint: Find all objects with value 71

int: num_objects;
set of int: OBJECTS = 1..num_objects;

% Each object has a value
array[OBJECTS] of var 0..1000: values;

% Constraint: extracted objects must have value 71
array[OBJECTS] of var bool: extracted;
constraint forall(i in OBJECTS)(
  extracted[i] <-> values[i] = 71
);

% Count extracted objects
var int: num_extracted = sum(i in OBJECTS)(extracted[i]);

% Solve
solve satisfy;

output [
  "Found \(num_extracted) objects with value 71\n",
  "Objects: \([i | i in OBJECTS where fix(extracted[i])])\n"
];
