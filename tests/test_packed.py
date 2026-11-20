import pytest
import torch

from bitladder.models import TinyCNN
from bitladder.packed import pack_model, read_packed, verify_packed_model
from bitladder.prune import apply_magnitude_pruning
from bitladder.quant import QuantConfig, apply_quantization


def test_packed_roundtrip_with_pruning_and_crc():
    torch.manual_seed(0)
    model = TinyCNN(width=8, bn=False)
    apply_quantization(model, QuantConfig(scheme="uniform", wbits=4, first_last_bits=None))
    apply_magnitude_pruning(model, 0.5, "conv2,fc")
    blob, report = pack_model(model)
    assert verify_packed_model(model, blob)
    packed = read_packed(blob)
    assert report["format"] == "BLP1" and report["total_bytes"] > report["payload_bytes"]
    conv2 = next(r for r in packed["layers"] if r["meta"]["name"] == "conv2")
    assert len(conv2["weight"]) == conv2["meta"]["nonzero_params"]
    assert conv2["mask"].sum() == conv2["meta"]["nonzero_params"]
    corrupted = blob[:-5] + bytes([blob[-5] ^ 1]) + blob[-4:]
    with pytest.raises(ValueError, match="CRC32"):
        read_packed(corrupted)


def test_packed_export_rejects_unfolded_batchnorm():
    with pytest.raises(ValueError, match="fold BatchNorm"):
        pack_model(TinyCNN(width=8, bn=True))
