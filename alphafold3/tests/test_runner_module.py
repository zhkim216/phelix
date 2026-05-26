from __future__ import annotations

import json

import pytest

from alphafold3 import run_alphafold
from alphafold3.common import folding_input


def test_af3_runner_template_chain_id_roundtrip():
    fold_input = folding_input.Input.from_json(
        json.dumps({
            "name": "template-chain-id",
            "modelSeeds": [1],
            "sequences": [{
                "protein": {
                    "id": "A",
                    "sequence": "ACD",
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [{
                        "mmcif": "data_template\n",
                        "queryIndices": [0, 1, 2],
                        "templateIndices": [5, 6, 7],
                        "templateChainId": "B",
                    }],
                },
            }],
            "dialect": folding_input.JSON_DIALECT,
            "version": folding_input.JSON_VERSION,
        })
    )

    assert fold_input.protein_chains[0].templates[0].template_chain_id == "B"
    roundtripped = folding_input.Input.from_json(fold_input.to_json())
    assert roundtripped.protein_chains[0].templates[0].template_chain_id == "B"
    assert roundtripped == fold_input


def test_af3_runner_ligand_template_conditioning_config():
    model_config = run_alphafold.make_model_config(
        ligand_protein_template_conditioning_mode=1,
        mask_template_sidechains=True,
        mask_template_sequence=True,
    )

    assert model_config.evoformer.template.ligand_protein_template_conditioning_mode == 1
    assert model_config.evoformer.template.mask_template_sidechains is True
    assert model_config.evoformer.template.mask_template_sequence is True


def test_af3_runner_ligand_template_conditioning_rejects_zero_templates():
    with pytest.raises(ValueError, match="max_templates must be > 0"):
        run_alphafold._validate_ligand_protein_template_conditioning_flags(1, 0)


def test_af3_runner_exposes_inprocess_eval_api():
    assert hasattr(run_alphafold, "make_model_config")
    assert hasattr(run_alphafold, "ModelRunner")
    assert hasattr(run_alphafold, "replace_db_dir")
    assert hasattr(run_alphafold, "process_fold_input")
