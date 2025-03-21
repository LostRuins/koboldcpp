#!/bin/bash
echo "Create Version File"
extracted_ver=$(cat koboldcpp.py | grep 'KcppVersion = ' | cut -d '"' -f2)
echo "Extracted Version: $extracted_ver"
vmajor=$(echo $extracted_ver | cut -d '.' -f1)
vminor=$(echo $extracted_ver | cut -d '.' -f2)
echo "Parsed Major Version: $vmajor"
echo "Parsed Minor Version: $vminor"
cp version_template.txt version.txt
sed "s/MYVER_MAJOR/$vmajor/g" version.txt > tempversion.txt && mv tempversion.txt version.txt
sed "s/MYVER_MINOR/$vminor/g" version.txt > tempversion.txt && mv tempversion.txt version.txt