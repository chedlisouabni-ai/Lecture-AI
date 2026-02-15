from setuptools import setup, find_packages

setup(
    name="lecture-ai",
    version="1.0.0",
    description="Multimodal Lecture Summarization System",
    author="Your Team",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'lecture-ai=main:main',
        ],
    },
)