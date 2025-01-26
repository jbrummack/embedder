{ pkgs ? import <nixpkgs> {} }:

#export PATH=$PATH:/nix/var/nix/profiles/default/bin/

pkgs.mkShell {
  buildInputs = [
    pkgs.rustc
    pkgs.cargo
    pkgs.bzip2
  ];
}
