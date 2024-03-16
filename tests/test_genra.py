#!/usr/bin/env python

"""Tests for `genra_py` package."""

import pytest


from genra.rax.skl.cls import GenRAPredClass
from genra.rax.skl.reg import GenRAPredValue

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
