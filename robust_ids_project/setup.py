from setuptools import find_packages, setup

setup(
    name="robust-ids",
    version="1.0.0",
    description="Robust intrusion detection under distribution shift",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.9.0",
        "PyYAML>=6.0.0",
        "joblib>=1.4.0",
        "scipy>=1.11.0",
    ],
    extras_require={
        "explain": ["shap>=0.46.0"],
    },
)

