#!/bin/bash
chmod +x "./kcpp_src/packaging/create_ver_file.sh"
. "./kcpp_src/packaging/create_ver_file.sh"
pyinstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all jinja2 --collect-all psutil --icon "./niko.ico" \
--add-data "./kcpp_adapters:./kcpp_adapters" \
--add-data "./koboldcpp.py:." \
--add-data "./kcpp_agent.py:." \
--add-data "./kcpp_src/json_to_gbnf.py:." \
--add-data "./LICENSE.md:."  \
--add-data "./MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md:." \
--add-data "./embd_res:./embd_res" \
--add-data "./koboldcpp_default.so:." \
--add-data "./koboldcpp_failsafe.so:." \
--add-data "./koboldcpp_noavx2.so:." \
--add-data "./koboldcpp_vulkan_failsafe.so:." \
--add-data "./koboldcpp_vulkan_noavx2.so:." \
--add-data "./koboldcpp_vulkan.so:." \
--version-file "./kcpp_src/packaging/version.txt" \
"./koboldcpp.py" -n "koboldcpp"
