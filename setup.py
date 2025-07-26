from setuptools import setup, find_packages

setup(
    name="project_curiosity",
    version="0.1.0",
    description="Learning concept–action mappings with LLM feedback",
    packages=find_packages(include=["project_curiosity*"]),
    install_requires=[
        "torch",
        "openai>=1.0.0",
    ],
    python_requires=">=3.8",
)
