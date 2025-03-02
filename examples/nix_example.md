# Guide to Nix for KoboldCpp

- KoboldCpp is available on Nixpkgs and can be installed by adding just
`koboldcpp` to your `environment.systemPackages` *(or it can also be placed
in `home.packages`)*.

## KoboldCpp Nix - CUDA Support

In order to enable NVIDIA CUDA support, you'll need to configure several
settings:

- Enable required options:

```nix
nixpkgs.config.allowUnfree = true;    # Allow proprietary software
nixpkgs.config.cudaSupport = true;    # Enable CUDA functionality
```

- Set your GPU architecture:

```nix
nixpkgs.config.cudaCapabilities = [ "sm_75" ];  # Example for RTX 2080
```

To find your GPU's architecture code:

1. Visit the [NVIDIA Architecture Guide](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
2. Locate your GPU Architecture
3. Use the corresponding `sm_XX` code in your configuration

## Hardware Support

- ✅ Vulkan: Enabled by default on Linux
- ✅ Metal: Enabled by default on macOS
- ❌ ROCm: Not currently available

## Example KoboldCpp Nix Configuration

```nix
nixpkgs.config = {
  allowUnfree = true;
  cudaSupport = true;
  cudaCapabilities = [ "sm_75" ];
};
environment.systemPackages = [ pkgs.koboldcpp ];
# If you're using home-manager to install KoboldCpp
# home.packages = [ pkgs.koboldcpp ];

# You can also just override koboldcpp to add your CUDA architecture:
# environment.systemPackages = [ (koboldcpp.override { cudaArches = ["sm_75"]; }) ]
# or
# home.packages = [ (koboldcpp.override { cudaArches = ["sm_75"]; }) ];
```

## KoboldCpp - Home Manager

The setup for Home Manager is the same as regular Nix, with one exception
regarding Home Manager's instance of nixpkgs. By default, Home Manager manages
its own isolated instance of nixpkgs, which has two implications:

1. You can keep your private Home Manager nixpkgs instance and simply repeat
your `nixpkgs.config` in home manager.
2. You can set `home-manager.useGlobalPkgs = true;` to copy your module
system's nixpkgs instance. This way, you only need to define it in your
`configuration.nix`, and Home Manager will "inherit" this configuration.

## Getting Help for KoboldCpp Nix

- If you face any issues with running KoboldCpp on Nix, please open an issue
[here](https://github.com/NixOS/nixpkgs/issues/new?assignees=&labels=0.kind%3A+bug&projects=&template=bug_report.md&title=)
