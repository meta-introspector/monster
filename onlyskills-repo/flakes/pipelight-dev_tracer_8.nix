{
  description = "pipelight-dev_tracer_8 by pipelight-dev";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs }: {
    packages.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.stdenv.mkDerivation {
      pname = "pipelight-dev_tracer_8";
      version = "1.0.0";
      src = ./.;
      
      buildPhase = ''
        echo "Building pipelight-dev_tracer_8..."
      '';
      
      installPhase = ''
        mkdir -p $out/bin
        echo "#!/bin/sh" > $out/bin/pipelight-dev_tracer_8
        echo "echo 'Executing pipelight-dev_tracer_8 by pipelight-dev'" >> $out/bin/pipelight-dev_tracer_8
        chmod +x $out/bin/pipelight-dev_tracer_8
      '';
    };
  };
}