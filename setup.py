#!/usr/bin/env python3
"""
Setup script for ASCII Art Converter
"""

from setuptools import setup, find_packages
import os

# Read the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ascii-art-converter',
    version='1.2.0',
    description='A powerful and flexible Python library for converting images to ASCII art, Braille art, and edge detection art',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/our0boros/ascii-art-converter',
    author='our0boros',
    author_email='our0boros@163.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Text Processing :: General',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='ascii art converter braille edge detection image processing',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'Pillow>=8.0.0',
        'numpy>=1.19.0,<2.0.0'
    ],
    extras_require={
        'advanced_edge_detection': [
            'scipy>=1.7.0',
            'scikit-image>=0.18.0,<0.22.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'ascii-art-converter=ascii_art_converter.cli:main',
            'ascii-interactive=ascii_art_converter.interactive:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/our0boros/ascii-art-converter/issues',
        'Source': 'https://github.com/our0boros/ascii-art-converter/',
    },
)
