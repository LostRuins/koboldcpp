import os
import sys

# Determine the folder where the .pyd files will be located
pyd_subdir = os.path.join(sys._MEIPASS, 'pyds')

# Add the subfolder to sys.path so Python can find the .pyd modules
print("Augmenting PYD directory...")
if os.path.isdir(pyd_subdir):
    sys.path.insert(0, pyd_subdir)