"""Meaningful checks for annotation parsing, leakage, sampling, and pooling."""
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from src.data_loader import flatten_sentence, clean_and_group, training_indices, LABELS


def test_nested_xml_offsets_and_no_markup():
    sentence = ET.fromstring('<sentence>A <xcope id="X"><cue type="negation">no</cue> '
                             'B</xcope>, C.</sentence>')
    text, spans = flatten_sentence(sentence)
    assert text == "A no B, C."
    assert {s["tag"]: text[s["start"]:s["end"]] for s in spans} == {
        "cue": "no", "xcope": "no B"}


def test_duplicate_documents_remain_connected_and_conflicts_removed():
    def row(doc, text, label="no_cue"):
        return {"id": doc+text, "doc_id":doc, "text":text, "label":label}
    rows = [row("a","same"), row("b"," SAME "), row("b","unique"),
            row("c","conflict"),row("d","conflict","negation_only")]
    clean, audit = clean_and_group(rows)
    assert len(clean) == 2
    assert len({r["group_id"] for r in clean}) == 1
    assert audit["conflicting_duplicate_rows_removed"] == 2


def test_nested_shared_training_subsets():
    rows = [{"id":str(i), "label":label, "split":"train"}
            for label in LABELS for i in range(15)]
    a, b = training_indices(rows, 5, 13), training_indices(rows, 10, 13)
    assert set(a) <= set(b)
    np.testing.assert_array_equal(a, training_indices(rows, 5, 13))
    with pytest.raises(ValueError):
        training_indices(rows, 16, 13)


def test_pooling_masks_and_truncated_span():
    import torch
    from src.feature_extraction import masked_mean, target_span_mean
    h = torch.tensor([[[100.,100.],[2.,4.],[4.,8.],[90.,90.],[999.,999.]]])
    att = torch.tensor([[1,1,1,1,0]])
    special = torch.tensor([[1,0,0,1,1]])
    assert torch.allclose(masked_mean(h,att,special),torch.tensor([[3.,6.]]))
    offsets = torch.tensor([[[0,0],[0,2],[3,5],[0,0],[0,0]]])
    assert torch.allclose(target_span_mean(h,offsets,att,[(3,5)]),h[:,2])
    with pytest.raises(ValueError):
        target_span_mean(h,offsets,att,[(3,7)])
