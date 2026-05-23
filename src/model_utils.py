import os
import sys
import torch
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def add_training_repo_to_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_root = os.path.dirname(current_dir)
    project_root = os.path.dirname(app_root)
    training_repo_path = os.path.join(project_root, "audio-deepfake-detection")

    if training_repo_path not in sys.path:
        sys.path.append(training_repo_path)


def build_model():
    add_training_repo_to_path()

    from AASIST3.model import aasist3

    model = aasist3.from_pretrained("MTUCI/AASIST3")

    return model


@st.cache_resource
def load_model(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    model = build_model()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    return model