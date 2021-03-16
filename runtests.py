#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" script for running all unit tests at once

TODO make this script obsolete
"""

import sys

import pytest


def main():
    sys.path.insert(0, "tests")
    return pytest.main()


if __name__ == "__main__":
    sys.exit(main())
