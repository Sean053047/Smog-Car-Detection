from pathlib import Path

from setuptools import setup, find_packages
import re

here = Path(__file__).parent.resolve()


# Get the long description from the README file
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


project_name = "yolov7"
setup(
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="yolov7",  # Required
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=get_property("__version__", project_name),  
    # Required to change the Verion number go to yolov7/__init__.py
    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description="A Python package of concise yolov7",  # Optional
    # url='*',
    author="Chien-Yao Wang | Rebuild by Yi-Xiang Yang",
    packages=find_packages(),
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.7, <4",
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python",
        "Pillow",
        "PyYAML",
        "requests",
        "scipy",
        "torch",
        "torchvision",
        "tqdm",
        "protobuf",
        "tensorboard",
        "pandas",
        "seaborn",
        "ipython",
        "psutil",
        "thop"
    ],  # Optional
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={
        "Bug Reports": "*/issues",
        "Source": "*",
    },
)
