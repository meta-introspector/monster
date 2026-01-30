{
  description = "pipelight-dev_parser_0 by pipelight-dev";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  
  outputs = { self, nixpkgs }: {
    packages.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.stdenv.mkDerivation {
      pname = "pipelight-dev_parser_0";
      version = "1.0.0";
      src = ./.;
      
      buildPhase = ''
        echo "Building pipelight-dev_parser_0..."
      '';
      
      installPhase = ''
        mkdir -p $out/bin
        echo "#!/bin/sh" > $out/bin/pipelight-dev_parser_0
        echo "echo 'Executing pipelight-dev_parser_0 by pipelight-dev'" >> $out/bin/pipelight-dev_parser_0
        chmod +x $out/bin/pipelight-dev_parser_0
      '';
    };
  };
}