import numpy as np

from atomworks.io.utils.sequence import (
    is_glycine,
    is_protein_unknown,
    is_purine,
    is_pyrimidine,
    is_standard_aa,
    is_standard_aa_not_glycine,
    is_unknown_nucleotide,
)


def test_is_pyrimidine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert (is_pyrimidine(sequence) == np.array(expected)).all()


def test_is_purine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert (is_purine(sequence) == np.array(expected)).all()


def test_is_unknown_nucleotide():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    assert (is_unknown_nucleotide(sequence) == np.array(expected)).all()


def test_is_protein():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_standard_aa(sequence) == np.array(expected)).all()


def test_is_glycine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_glycine(sequence) == np.array(expected)).all()


def test_is_protein_not_glycine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_standard_aa_not_glycine(sequence) == np.array(expected)).all()


def test_is_protein_unknown():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC",
        "DG",
        "DT",
        "UNK",
        "A",
        "C",
        "G",
        "U",
        "DN",
        "N",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_protein_unknown(sequence) == np.array(expected)).all()
