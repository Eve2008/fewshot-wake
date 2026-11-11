# Copied unchanged from NexaRoom (nexaroom/voice/features.py, commit be63f3e, October 2026).
# Only the import of FEATURE_DIM/FRAME_SECONDS is package-relative, as in the original.
"""Neural acoustic features for few-shot wake detection.

Uses the two small, shared ONNX models published with openWakeWord:
``melspectrogram.onnx`` (32-band log-mel, 10 ms hop) and
``embedding_model.onnx`` (Google's ``speech_embedding``: 96 values every
80 ms, each summarising the previous ~0.76 s of audio). The models are
language independent and are not tied to any particular wake phrase, so the
same features represent Persian words, English words, or non-speech sounds
such as a whistle or a finger snap.

The streaming code path is used for both live audio and enrollment clips, so
templates and live features are produced in exactly the same way.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

CHUNK_SAMPLES = 1280            # 80 ms at 16 kHz -> one embedding
MEL_CONTEXT_SAMPLES = 480       # extra history so 1760 samples yield 8 mel frames
MEL_FRAMES_PER_CHUNK = 8
EMBEDDING_WINDOW = 76           # mel frames per embedding
EMBEDDING_DIM = 96
MEL_BANDS = 32
# Each feature frame = neural embedding (what was said, ~0.76 s context)
# + mean log-mel of the newest 80 ms (pitch/timbre trajectory). The mel part
# separates non-speech sounds the speech embedding barely tells apart,
# e.g. a rising versus a falling whistle.
FEATURE_DIM = EMBEDDING_DIM + MEL_BANDS
FEATURE_VERSION = 3
FRAME_SECONDS = CHUNK_SAMPLES / 16000

MEL_MODEL = "melspectrogram.onnx"
EMBEDDING_MODEL = "embedding_model.onnx"


class FeaturesUnavailable(RuntimeError):
    pass


class EmbeddingModels:
    """Lazily loaded ONNX sessions shared by all feature streams."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._lock = RLock()
        self._mel: Any = None
        self._embedding: Any = None
        self._error: str | None = None

    @property
    def mel_path(self) -> Path:
        return self.directory / MEL_MODEL

    @property
    def embedding_path(self) -> Path:
        return self.directory / EMBEDDING_MODEL

    def files_ready(self) -> bool:
        return self.mel_path.is_file() and self.embedding_path.is_file()

    def status(self) -> dict[str, Any]:
        try:
            import importlib.util

            runtime_ready = importlib.util.find_spec("onnxruntime") is not None
        except (ImportError, ValueError):
            runtime_ready = False
        files = self.files_ready()
        reason = None
        if not files:
            reason = "مدل‌های ویژگی صوتی wake دانلود نشده‌اند."
        elif not runtime_ready:
            reason = "بستهٔ onnxruntime نصب نیست."
        elif self._error:
            reason = self._error
        return {"ready": reason is None, "files": files, "runtime": runtime_ready, "reason": reason}

    def reload(self) -> None:
        with self._lock:
            self._mel = None
            self._embedding = None
            self._error = None

    def load(self) -> None:
        with self._lock:
            if self._mel is not None and self._embedding is not None:
                return
            status = self.status()
            if not status["ready"]:
                raise FeaturesUnavailable(status["reason"])
            try:
                import onnxruntime as ort  # type: ignore[import-not-found]

                options = ort.SessionOptions()
                options.inter_op_num_threads = 1
                options.intra_op_num_threads = 1
                options.log_severity_level = 3
                providers = ["CPUExecutionProvider"]
                self._mel = ort.InferenceSession(str(self.mel_path), sess_options=options, providers=providers)
                self._embedding = ort.InferenceSession(str(self.embedding_path), sess_options=options,
                                                       providers=providers)
            except Exception as exc:
                self._mel = None
                self._embedding = None
                self._error = "مدل‌های ویژگی صوتی wake بارگذاری نشدند."
                raise FeaturesUnavailable(self._error) from exc

    def melspectrogram(self, samples: Any) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        self.load()
        data = np.asarray(samples, dtype=np.float32).reshape(1, -1)
        output = self._mel.run(None, {"input": data})[0]
        # Same transform openWakeWord applies to match the TF reference model.
        return np.squeeze(output, axis=(0, 1)) / 10.0 + 2.0

    def embed(self, windows: Any) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        self.load()
        batch = np.asarray(windows, dtype=np.float32).reshape(-1, EMBEDDING_WINDOW, 32, 1)
        output = self._embedding.run(None, {"input_1": batch})[0]
        return output.reshape(batch.shape[0], EMBEDDING_DIM)

    def stream(self) -> "FeatureStream":
        return FeatureStream(self)

    def clip_features(self, pcm: bytes) -> tuple[Any, Any]:
        """Return (features[T, FEATURE_DIM], end_sample_positions[T]) for a whole clip."""

        import numpy as np  # type: ignore[import-not-found]

        stream = self.stream()
        stream.push(pcm, emit=False)
        if not stream.history:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)
        values, ends = zip(*stream.history)
        return np.vstack(values).astype(np.float32), np.asarray(ends, dtype=np.int64)


class FeatureStream:
    """Incremental features: push PCM, receive one feature frame per 80 ms."""

    def __init__(self, models: EmbeddingModels) -> None:
        import numpy as np  # type: ignore[import-not-found]

        self._models = models
        self._np = np
        self._raw = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(MEL_CONTEXT_SAMPLES, dtype=np.float32)
        self._mel = np.ones((EMBEDDING_WINDOW, 32), dtype=np.float32)
        self._consumed = 0
        self.history: list[tuple[Any, int]] = []

    def reset(self) -> None:
        self.__init__(self._models)  # type: ignore[misc]

    def push(self, pcm: bytes, *, emit: bool = True) -> list[Any]:
        """Add PCM. Returns new embeddings; with emit=False they are only kept in history."""

        np = self._np
        if pcm:
            values = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32)
            self._raw = np.concatenate((self._raw, values)) if self._raw.size else values
        produced = []
        while self._raw.size >= CHUNK_SAMPLES:
            chunk = self._raw[:CHUNK_SAMPLES]
            self._raw = self._raw[CHUNK_SAMPLES:]
            mel_input = np.concatenate((self._context, chunk))
            self._context = mel_input[-MEL_CONTEXT_SAMPLES:]
            mel = self._models.melspectrogram(mel_input)[-MEL_FRAMES_PER_CHUNK:]
            self._mel = np.vstack((self._mel, mel))[-EMBEDDING_WINDOW:]
            self._consumed += CHUNK_SAMPLES
            embedding = self._models.embed(self._mel[None, :, :])[0]
            embedding = np.concatenate((embedding, mel.mean(axis=0))).astype(np.float32)
            if emit:
                produced.append(embedding)
            else:
                self.history.append((embedding, self._consumed))
        return produced
