import numpy as np
import soundfile as sf
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 64600
PRE_EMPHASIS_COEF = 0.97


def load_audio(file_path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio menggunakan soundfile, bukan torchaudio.load,
    agar tidak membutuhkan TorchCodec di Windows.

    Return:
        waveform: Tensor [channels, samples]
        sample_rate: int
    """
    audio_np, sample_rate = sf.read(file_path, dtype="float32")

    waveform = torch.from_numpy(audio_np)

    # mono: [samples] -> [1, samples]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # stereo/multichannel: [samples, channels] -> mean -> [1, samples]
    elif waveform.ndim == 2:
        waveform = waveform.mean(dim=1).unsqueeze(0)

    else:
        raise ValueError(f"Unsupported audio shape: {waveform.shape}")

    return waveform, sample_rate


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def resample_audio(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            original_sample_rate,
            target_sample_rate,
        )

    return waveform


def apply_pre_emphasis(
    waveform: torch.Tensor,
    coef: float = PRE_EMPHASIS_COEF,
) -> torch.Tensor:
    waveform = waveform.squeeze(0)

    if waveform.numel() <= 1:
        return waveform.unsqueeze(0)

    emphasized = torch.cat(
        [
            waveform[:1],
            waveform[1:] - coef * waveform[:-1],
        ]
    )

    return emphasized.unsqueeze(0)


def pad_or_truncate(
    waveform: torch.Tensor,
    target_num_samples: int = TARGET_NUM_SAMPLES,
) -> torch.Tensor:
    waveform = waveform.squeeze(0)
    num_samples = waveform.shape[0]

    if num_samples > target_num_samples:
        waveform = waveform[:target_num_samples]
    elif num_samples < target_num_samples:
        pad_length = target_num_samples - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    return waveform


def preprocess_audio(file_path: str) -> torch.Tensor:
    """
    Pipeline preprocessing untuk inference.

    Output:
        Tensor [1, TARGET_NUM_SAMPLES]
    """
    waveform, sample_rate = load_audio(file_path)

    waveform = convert_to_mono(waveform)
    waveform = resample_audio(waveform, sample_rate)
    waveform = apply_pre_emphasis(waveform)
    waveform = pad_or_truncate(waveform)

    return waveform.unsqueeze(0)