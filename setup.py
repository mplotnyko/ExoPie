from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="exopie",
    version="1.2.3",
    author="Mykhaylo Plotnykov",
    author_email="mykhaylo.plotnykov@mail.utoronto.ca",
    description="A package for finding exoplanet interiors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mplotnyko/exopie/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"exopie": ["Data/*"]},
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy"
    ],
    extras_require={
        "optional": ["corner"]
    },
)
