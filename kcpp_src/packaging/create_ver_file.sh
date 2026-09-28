#!/bin/bash
echo "Create Version File"
version_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$version_dir/../.." && pwd)"
extracted_ver=$(grep 'KcppVersion = ' "$repo_root/koboldcpp.py" | cut -d '"' -f2)
echo "Extracted Version: $extracted_ver"
vmajor=$(echo $extracted_ver | cut -d '.' -f1)
vminor=$(echo $extracted_ver | cut -d '.' -f2)
echo "Parsed Major Version: $vmajor"
echo "Parsed Minor Version: $vminor"
cp "$version_dir/version_template.txt" "$version_dir/version.txt"
sed "s/MYVER_MAJOR/$vmajor/g" "$version_dir/version.txt" > "$version_dir/tempversion.txt" && mv "$version_dir/tempversion.txt" "$version_dir/version.txt"
sed "s/MYVER_MINOR/$vminor/g" "$version_dir/version.txt" > "$version_dir/tempversion.txt" && mv "$version_dir/tempversion.txt" "$version_dir/version.txt"
