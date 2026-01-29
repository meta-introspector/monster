{ pkgs ? import <nixpkgs> {} }:

let
  # GAP with packages
  gap-with-packages = pkgs.gap.override {
    packageSet = pkgs.gap.packageSet.override {
      packages = with pkgs.gap.packageSet; [
        atlasrep
        ctbllib
      ];
    };
  };

in pkgs.mkShell {
  name = "monster-experiments";
  
  buildInputs = [
    gap-with-packages
    pkgs.jq
    pkgs.python3
    pkgs.python3Packages.pandas
    pkgs.python3Packages.pyarrow
  ];
  
  shellHook = ''
    export EXPERIMENT_DIR="$PWD/experiments"
    mkdir -p "$EXPERIMENT_DIR"
    
    echo "🧪 Monster Group Experiments"
    echo "============================"
    echo ""
    echo "Experiments will be saved to: $EXPERIMENT_DIR"
    echo ""
    echo "Available experiments:"
    echo "  ./experiments/01_monster_order.sh"
    echo "  ./experiments/02_conjugacy_classes.sh"
    echo "  ./experiments/03_character_table.sh"
    echo "  ./experiments/04_backtrack_test.sh"
  '';
}
