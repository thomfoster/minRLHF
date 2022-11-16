from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='minRLHF',
    version='0.0.0',
    url='https://github.com/thomfoster/minRLHF',
    author='Thomas Foster',
    author_email='fosterthom16@gmail.com',
    description="Minimal implementation to train Karpathy's minGPT using PPO on human feedback. For educational purposes.",
    packages=find_packages(),    
    install_requires=required,
)