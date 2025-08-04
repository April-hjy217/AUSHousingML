import pytest
import pandas as pd
from src.train import extract_num

def test_extract_num_int():
    assert extract_num("Building 123m2") == 123

def test_extract_num_float():
    assert extract_num("80.5 sqm") == 80.5

def test_extract_num_none():
    assert extract_num(None) is None
    assert extract_num("No number") is None


