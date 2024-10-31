# setup.py
import platform
from setuptools import setup

# Common dependencies
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# OS-specific dependencies
if platform.system() == "Linux" or platform.system() == "Darwin":
    install_requires.append("posix-ipc==1.1.1")
elif platform.system() == "Windows":
    install_requires.extend([
        "win32api==1.0.0",
        "win32pipe==1.0.0",
        "win32process==1.0.0"
    ])

setup(
    name="simulateTracks",
    version="0.1",
    install_requires=install_requires,
    # other setup parameters
)