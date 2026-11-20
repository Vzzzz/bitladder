import json
from types import SimpleNamespace

import numpy as np
import pytest

from audio.kws.stream import WakeWordStreamer


def zero_features(_x):
    return np.zeros((49, 40), dtype=np.float32)


class DummySession:
    def __init__(self, logits):
        self.logits = np.asarray(logits, dtype=np.float32)[None]
        self.last_x = None

    def get_inputs(self):
        return [SimpleNamespace(name="features")]

    def run(self, _outputs, inputs):
        self.last_x = inputs["features"]
        return [self.logits]


def make_streamer(tmp_path, session, feature_fn=None):
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"name": "test-model", "classes": ["yes", "no", "kostyan"]}))
    calib = tmp_path / "calib.npz"
    np.savez(calib, mean=np.float32(0.0), std=np.float32(1.0))
    return WakeWordStreamer("unused.onnx", str(run), str(calib), "kostyan", threshold=0.8,
                            smooth_windows=3, refractory_s=1.0, session=session, feature_fn=feature_fn)


def test_streaming_events_are_invariant_to_pcm_chunks(tmp_path):
    pcm = np.zeros(24000, dtype=np.int16)
    feature_fn = zero_features
    whole = make_streamer(tmp_path, DummySession([0.0, 0.0, 10.0]), feature_fn)
    chunked = make_streamer(tmp_path, DummySession([0.0, 0.0, 10.0]), feature_fn)
    events = whole.accept_pcm(pcm)
    chunk_events = []
    for chunk in np.array_split(pcm, 37):
        chunk_events.extend(chunked.accept_pcm(chunk))
    assert events == chunk_events
    assert len(events) == 1
    assert events[0]["event"] == "wake" and events[0]["word"] == "kostyan"
    assert events[0]["model"] == "test-model"
    assert events[0]["time_s"] == pytest.approx(1.2)
    assert events[0]["confidence"] == pytest.approx(1.0, abs=1e-4)


def test_streamer_uses_refractory_and_reset(tmp_path):
    pcm = np.zeros(16000, dtype=np.float32)
    feature_fn = zero_features
    stream = make_streamer(tmp_path, DummySession([0.0, 0.0, 10.0]), feature_fn)
    stream.refractory_s = 10.0
    assert not stream.accept_pcm(pcm)
    assert len(stream.accept_pcm(pcm)) == 1
    assert not stream.accept_pcm(pcm[:8000])  # inside the extended refractory interval
    stream.reset()
    assert not stream.accept_pcm(pcm)  # reset also clears the smoothing history
    assert len(stream.accept_pcm(pcm)) == 1


def test_streamer_real_logmel_shape_and_int16_bytes(tmp_path):
    session = DummySession([0.0, 0.0, 0.0])
    stream = make_streamer(tmp_path, session)
    assert not stream.accept_pcm(np.zeros(16000, dtype=np.int16).tobytes())
    assert session.last_x.shape == (1, 1, 49, 40)
    assert session.last_x.dtype == np.float32
