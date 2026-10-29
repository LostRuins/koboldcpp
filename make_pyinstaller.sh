#!/bin/bash
chmod +x "./kcpp_src/packaging/create_ver_file.sh"
. "./kcpp_src/packaging/create_ver_file.sh"
pyinstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all jinja2 --collect-all psutil --icon "./niko.ico" \
--add-data "./kcpp_adapters:./kcpp_adapters" \
--add-data "./koboldcpp.py:." \
--add-data "./kcpp_agent.py:." \
--add-data "./kcpp_src/json_to_gbnf.py:." \
--add-data "./LICENSE.md:."  \
--add-data './ggml/src/GGML_LICENSE:./vendor_licenses'  \
--add-data './otherarch/acestep/ACESTEP_LICENSE:./vendor_licenses'  \
--add-data './otherarch/audiocpp/upstream/AUDIOCPP_LICENSE:./vendor_licenses'  \
--add-data './otherarch/qwen3tts/QWEN3TTS_LICENSE:./vendor_licenses'  \
--add-data './otherarch/sdcpp/SDCPP_LICENSE:./vendor_licenses'  \
--add-data './otherarch/ttscpp/TTSCPP_LICENSE:./vendor_licenses'  \
--add-data './src/LLAMACPP_LICENSE:./vendor_licenses'  \
--add-data "./embd_res:./embd_res" \
--add-data "./koboldcpp_default.so:." \
--add-data "./koboldcpp_failsafe.so:." \
--add-data "./koboldcpp_noavx2.so:." \
--add-data "./koboldcpp_vulkan_failsafe.so:." \
--add-data "./koboldcpp_vulkan_noavx2.so:." \
--add-data "./koboldcpp_vulkan.so:." \
--version-file "./kcpp_src/packaging/version.txt" \
"./koboldcpp.py" -n "koboldcpp"
