from setuptools import find_packages, setup

setup(
    name='genra',
    packages=find_packages(),
    version='0.1.3',
    install_requires=['numpy', 'scipy', 'scikit-learn==1.0.1', 'lxml'],
    description='Generalised Read Across (GenRA) in Python',
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
