
## how to use test_interactions.py

- remove old ground-truth file: `rm pypolygames/tests/data/Hex11.txt`

- run tests to generate a new ground-truth file: `pytest internal pypolygames --durations=10 --verbose -x`

- manually check that the generated file is correct: `cat pypolygames/tests/data/Hex11.txt`

- check that all tests pass: `pytest internal pypolygames --durations=10 --verbose`

