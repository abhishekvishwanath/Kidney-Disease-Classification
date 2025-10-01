from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """Reads the requirements from a file and returns them as a list."""

    requirements_list:List[str] = []
    try:
        with open("requirements.txt", 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!="-e .":
                    requirements_list.append(requirement)
        
    except FileNotFoundError:
        print(f"Error: The file requirements.txt was not found.")
    
    return requirements_list

setup(
    name="kidney_disease_classification",
    version="0.0.1",
    author="Abhishek Vishwanath",
    packages=find_packages(),
    install_requires=get_requirements(),
)



