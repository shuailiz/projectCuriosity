from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlp_question_generator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A MLP model for token reasoning and question generation with ChatGPT verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlp_question_generator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "transformers>=4.18.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
    ],
)
