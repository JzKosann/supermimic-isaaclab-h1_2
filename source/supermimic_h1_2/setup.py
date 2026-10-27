"""Installation script for the 'supermimic_h1_2' Isaac Lab extension."""

import os

import toml
from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "huggingface_hub",  # scripts/download_assets.py
]

setup(
    name="supermimic_h1_2",
    packages=["supermimic_h1_2", "supermimic_h1_2.mdp", "supermimic_h1_2.tasks"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
