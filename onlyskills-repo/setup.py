from setuptools import setup, find_packages

setup(
    name='onlyskills-zkerdaprologml',
    version='1.0.0',
    description='Zero-Knowledge 71-Shard Skill Registry for AI Agents',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='onlyskills.com',
    author_email='hello@onlyskills.com',
    url='https://onlyskills.com',
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.0',
        'rdflib>=6.0.0',
        'sparqlwrapper>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'onlyskills=onlyskills.server:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    python_requires='>=3.8',
    keywords='zkerdaprologml zero-knowledge skills ai-agents monster-group',
    project_urls={
        'Source': 'https://github.com/onlyskills/zkerdaprologml',
        'Documentation': 'https://docs.onlyskills.com',
    },
)
