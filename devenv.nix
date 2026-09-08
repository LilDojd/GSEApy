{ pkgs, lib, ... }:

{
  packages = [
    pkgs.python3
    pkgs.uv
  ]
  ++ lib.optionals pkgs.stdenv.isLinux [
    pkgs.linuxPackages.perf
    pkgs.valgrind
    pkgs.time
  ];

  languages.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
  };

  env.PYO3_PYTHON = "${pkgs.python3}/bin/python3";
  env.UV_PYTHON = "${pkgs.python3}/bin/python3";

  enterTest = ''
    cargo test --locked
    cargo test --locked --features simd
    cargo test --locked --features python
    cargo test --locked --features python,simd
  '';
}
