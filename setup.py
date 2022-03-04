#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="TUM Department of Electrical and Computer Engineering - Chair of Electronic Design Automation",
    author_email='philipp.van-kempen@tum.de',
    python_requires='>=3.7',
    classifiers=[  # TODO
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This project contains research code related to the deployment of inference or learning applications on tiny micro-controllers.",
    entry_points={
        'console_scripts': [
            'mlonmcu=mlonmcu.cli.main:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='mlonmcu',
    name='mlonmcu',
    packages=find_packages(include=['mlonmcu', 'mlonmcu.*']),
    package_data={'mlonmcu': ['../templates/*.j2']},
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tum-ei-eda/mlonmcu',
    version='0.1.0',
    zip_safe=False,
)
