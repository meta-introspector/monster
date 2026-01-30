{
  description = "pipelight-dev_compiler_1 by pipelight-dev";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs }: {
    packages.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.stdenv.mkDerivation {
      pname = "pipelight-dev_compiler_1";
      version = "1.0.0";
      src = ./.;
      
      buildPhase = ''
        echo "Building pipelight-dev_compiler_1..."
      '';
      
      installPhase = ''
        mkdir -p $out/bin
        echo "#!/bin/sh" > $out/bin/pipelight-dev_compiler_1
        echo "echo 'Executing pipelight-dev_compiler_1 by pipelight-dev'" >> $out/bin/pipelight-dev_compiler_1
        chmod +x $out/bin/pipelight-dev_compiler_1
      '';
    };
  };
}