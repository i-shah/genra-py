from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='genra',
    packages=find_packages(),
    version='0.2.3',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'lxml'],
    description='Generalised Read Across (GenRA) in Python',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/i-shah/genra-py",
    author='Imran Shah',
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'],
        keywords = 'genra'
)
