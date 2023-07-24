#!/bin/bash

# if this script fails make sure you've installed everything in requirements-test.txt

python3 -m coverage run -m pytest

python3 -m coverage report -i src/*.py

rm .coverage