from setuptools import setup, find_packages

setup(
    name="warp-alignment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "trl",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "pytest",
    ],
    author="Данила Катальшов",
    author_email="katalshovd@gmai.com",
    description="Реализация и анализ метода WARP для алаймента языковых моделей",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ADanMan/warp-alignment",
)