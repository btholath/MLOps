"""Purpose of setup.py
Role	Description
Package Metadata	Defines name, version, author, license, description, etc.
Dependency Management	Lists required packages via install_requires
Build & Distribution	Allows creation of distributable formats like .tar.gz or .whl
Installable Script	Enables installation with pip install . or python setup.py install
Entry Points (CLI)	Defines command-line scripts using entry_points
Version Control	Can tie version to __init__.py or Git tags


Common Usage
🔹 Installing Locally
pip install .

# or
python setup.py install

🔹 Building Distributables
python setup.py sdist bdist_wheel


## Python virtual environment
bijut@b:~/aws_apps/MLOps$ cd ..
bijut@b:~/aws_apps$ source .venv/bin/activate
(.venv) bijut@b:~/aws_apps$ cd MLOps/01_etl_pipeline/
(.venv) bijut@b:~/aws_apps/MLOps/01_etl_pipeline$
"""

from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
  """
  """
  requirement_list:List[str] = []
  try:
    with open('requirements.txt','r') as file:
      # read lines from the file
      lines = file.readlines()
      # process each line
      for line in lines:
        requirement=line.strip()
        # ignore empty lines and -e .
        if requirement and requirement != '-e .':
          requirement_list.append(requirement) 
        
  except FileNotFoundError:
    print("requirements.txt is missing!")
  
  return requirement_list  
  
print(get_requirements())    

setup(
    name='NetworkSecurity',
    version='0.1.0',
    description='A sample ETL pipeline project',
    author='Biju Tholath',
    author_email='biju.tholath@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
    entry_points={
        'console_scripts': [
            'run_pipeline=etl_pipeline.main:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3'
        
    ],
    license='License :: OSI Approved :: MIT License',
    python_requires='>=3.8',
)
