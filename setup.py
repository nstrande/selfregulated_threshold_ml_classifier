from setuptools import setup, find_packages

setup(
    name="selfregulated_threshold_ml_classifier",
    version="0.0.1",
    author="Niller",
    description="Functional example of how to use the wrapper that self-regulates the threshold with respect to either recall or precision.",
    packages=find_packages(),
    python_requires=">=3.8",
)
