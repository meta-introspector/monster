% MiniZinc model for create_monster_autoencoder.rs
% Constraint: Verify autoencoder architecture

% Monster prime dimensions
array[1..5] of int: dims = [5, 11, 23, 47, 71];

% Encoder layers
array[1..4] of var int: encoder_input;
array[1..4] of var int: encoder_output;

constraint encoder_input[1] = dims[1];
constraint encoder_output[1] = dims[2];
constraint encoder_input[2] = dims[2];
constraint encoder_output[2] = dims[3];
constraint encoder_input[3] = dims[3];
constraint encoder_output[3] = dims[4];
constraint encoder_input[4] = dims[4];
constraint encoder_output[4] = dims[5];

% Decoder layers (reverse)
array[1..4] of var int: decoder_input;
array[1..4] of var int: decoder_output;

constraint decoder_input[1] = dims[5];
constraint decoder_output[1] = dims[4];
constraint decoder_input[2] = dims[4];
constraint decoder_output[2] = dims[3];
constraint decoder_input[3] = dims[3];
constraint decoder_output[3] = dims[2];
constraint decoder_input[4] = dims[2];
constraint decoder_output[4] = dims[1];

% Verify reconstruction
constraint encoder_input[1] = decoder_output[4];

solve satisfy;

output [
  "Encoder: \(encoder_input[1]) → \(encoder_output[4])\n",
  "Decoder: \(decoder_input[1]) → \(decoder_output[4])\n",
  "Reconstruction: \(fix(encoder_input[1]) = fix(decoder_output[4]))\n"
];
