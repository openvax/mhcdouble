# Copyright (c) 2014. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import os
import re

from setuptools import setup, find_packages

readme_dir = os.path.dirname(__file__)
readme_filename = os.path.join(readme_dir, 'README.md')

try:
    with open(readme_filename, 'r') as f:
        readme = f.read()
    # create README.rst for deploying on Travis
    rst_readme_filename = readme_filename.replace(".md", ".rst")
    with open(rst_readme_filename, "w"):
        f.write(readme)
except:
    readme = ""

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    print("Conversion of README from MD to reStructuredText failed")

with open('mhc2/__init__.py', 'r') as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(),
        re.MULTILINE).group(1)

if not version:
    raise RuntimeError('Cannot find version information')

if __name__ == '__main__':
    setup(
        name='mhc2',
        version=version,
        description="Class II MHC binding and antigen processing prediction",
        author="Alex Rubinsteyn",
        author_email="alex@hammerlab.org",
        url="https://github.com/hammerlab/mhc2",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
        ],
        install_requires=[
            'numpy',
            'pandas',
            'ujson',
            'scikit-learn',
            'pepnet',
        ],
        long_description=readme,
        packages=find_packages(exclude="test"),
        entry_points={
            'console_scripts': [
                'mhc2-predict=mhc2.cli.predict:main',
                'mhc2-train-convolutional=mhc2.cli.train:main',
                'mhc2-assembly=mhc2.cli.assembly:main',
                'mhc2-binding-cores=mhc2.cli.binding_cores:main',
                'mhc2-generate-decoys=mhc2.cli.generate_decoys:main',
                'mhc2-generate-nested-decoys=mhc2.cli.generate_nested_decoys:main',
                'mhc2-list-alleles=mhc2.cli.list_alleles:main',
                'mhc2-train-fixed-length=mhc2.cli.train_fixed_length:main',
                'mhc2-predict-netmhciipan=mhc2.cli.predict_netmhciipan:main',
            ]
        }
    )
