genra-py
==============================

Generalised Read Across (GenRA) in Python

Project Organization
------------
No spaces in filenames!

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data from public domain sources.
    │   └─ shah-2016       <- https://doi.org/10.1016/j.yrtph.2016.05.008
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks (XXX-UU-DDDDDD.ipynb). Convention:- 
    |   |                     XXX = numeric sequence 
    |   |                     UU  = user initials
    |   |                     DDDDDDD = descriptive string 
    |   ├─is               <- Imran Shah
    |   ├─gp               <- Grace Patlewicz
    |   ├─tt               <- Tia Tate
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── condaenv.yaml      <- The spec for creating a conda environment. Generated using:
    |                          conda env export > condaenv.yml
    │                         Can create environment using:
    |                          conda env create -f condaenv.yml
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |
    ├── tools              <- GenRA command-line tools (installed in /usr/local/bin) that depend on src 
    └── src                <- Source code for use in this project.
        │
        └─genra          
            ├─db           <- Database access and etl
            ├─data         <- Data preparation / manipulation
            |  └─chm       <- chemical structure and physchem data processing 
            |  └─bio       <- bioactivity data processing             
            |  └─tox       <- toxicity data processing             
            ├─rax          <- Read Across prediction
            |  └─skl       <- Standalone based on scikit-learn
            |  └─srv       <- Server based on mongodb 
            ├─viz          <- Visualization 
            ├─utl          <- Utilities
            
           
Built using cookiecutter 