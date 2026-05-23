import torch

from src.audio_utils import preprocess_audio
from src.model_utils import DEVICE


CLASS_NAMES = {
    0: "Spoof / Deepfake",
    1: "Bona Fide / Asli"
}


def predict_audio(model, audio_path: str) -> dict:
    """
    Menjalankan inference pada satu file audio.

    Return:
        {
            "label": str,
            "class_id": int,
            "confidence": float,
            "probabilities": list[float]
        }
    """
    waveform = preprocess_audio(audio_path)
    waveform = waveform.to(DEVICE)

    with torch.no_grad():
        outputs = model(waveform)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_id = int(predicted_class.item())
    confidence_value = float(confidence.item()) * 100

    return {
        "label": CLASS_NAMES.get(class_id, f"Class {class_id}"),
        "class_id": class_id,
        "confidence": confidence_value,
        "probabilities": probabilities.squeeze(0).cpu().tolist()
    }