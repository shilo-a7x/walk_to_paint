"""Test that multiprocess encode_walks produces bit-exact results vs single-process."""
import numpy as np
import pytest

from src.data.tokenizer import Tokenizer
from src.data.prepare_data import encode_walks, SplitID


def _make_fixtures():
    edges = [
        (0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1),
        (0, 2, 0), (1, 3, 1), (2, 0, 0), (3, 1, 1),
    ]
    walks = [
        ["N_0", "E_0", "N_1", "E_1", "N_2", "E_0", "N_3"],
        ["N_1", "E_1", "N_3", "E_1", "N_1"],
        ["N_2", "E_0", "N_0", "E_0", "N_2"],
        ["N_3", "E_1", "N_1", "E_0", "N_0", "E_1", "N_3"],
        ["N_0", "E_0", "N_1"],
        ["N_2", "E_0", "N_3", "E_1", "N_1", "E_0", "N_0"],
        ["N_1", "E_0", "N_0"],
        ["N_3", "E_0", "N_2"],
    ] * 50  # 400 walks — enough to exercise chunking with several workers

    tok = Tokenizer()
    tok.fit(walks, edges)

    train_set = {(0, 1, 0), (1, 2, 1), (2, 3, 0)}
    mask_set  = {(3, 0, 1)}
    val_set   = {(0, 2, 0)}
    test_set  = {(1, 3, 1), (2, 0, 0), (3, 1, 1)}

    return walks, tok, edges, train_set, mask_set, val_set, test_set


@pytest.mark.parametrize("num_workers", [2, 4, 8])
def test_multiprocess_encode_matches_single(num_workers):
    walks, tok, edges, train_set, mask_set, val_set, test_set = _make_fixtures()

    single_ids, single_splits, single_eids = encode_walks(
        walks, tok, edges, train_set, mask_set, val_set, test_set,
        num_workers=1,
    )
    multi_ids, multi_splits, multi_eids = encode_walks(
        walks, tok, edges, train_set, mask_set, val_set, test_set,
        num_workers=num_workers,
    )

    assert len(single_ids) == len(multi_ids) == len(walks)
    for i in range(len(walks)):
        assert np.array_equal(single_ids[i],    multi_ids[i]),    f"input_ids mismatch at walk {i}"
        assert np.array_equal(single_splits[i], multi_splits[i]), f"split_mask mismatch at walk {i}"
        assert np.array_equal(single_eids[i],   multi_eids[i]),   f"edge_ids mismatch at walk {i}"
