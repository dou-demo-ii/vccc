import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
from src.model_utils import load_model
from src.predict import predict_audio

CHECKPOINT_DIR = "checkpoints"
SUPPORTED_CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}
SR = 16000
MAX_LEN = 64600


def discover_checkpoints(checkpoint_dir: str = CHECKPOINT_DIR) -> list[str]:
    """
    Find all supported model checkpoint files inside checkpoint_dir.

    Nested folders are supported, for example:
      checkpoints/aasist3_kan/best_e8.pt
      checkpoints/aasist3_baseline/best_e12.pt
    """
    base_dir = Path(checkpoint_dir)

    if not base_dir.exists():
        return []

    checkpoints = [
        str(path)
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_CHECKPOINT_EXTENSIONS
    ]

    return sorted(checkpoints)


def format_model_name(checkpoint_path: str) -> str:
    """Create a readable model label from its checkpoint path."""
    path = Path(checkpoint_path)

    try:
        relative_path = path.relative_to(CHECKPOINT_DIR)
    except ValueError:
        relative_path = path

    return str(relative_path.with_suffix("")).replace("\\", " / ")


@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Load and cache a model checkpoint.

    Switching back to a previously selected checkpoint is faster because
    Streamlit reuses the already-loaded model instance.
    """
    return load_model(checkpoint_path)


# ═══════════════════════════════════════════════════════════════════════════ #
#  XAI — Audio loading
# ═══════════════════════════════════════════════════════════════════════════ #


def load_audio_for_xai(audio_path: str) -> tuple[np.ndarray, int]:
    """Load any supported format (FLAC/WAV/MP3) to mono 16 kHz numpy array."""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    return y, SR


# ═══════════════════════════════════════════════════════════════════════════ #
#  XAI — Plot helpers
# ═══════════════════════════════════════════════════════════════════════════ #

_BG = "#0E1117"
_FG = "white"
_GRID = "#2A2A3A"
_ACCENT = "#4FC3F7"  # light blue
_DANGER = "#FF6B6B"  # red for deepfake markers


def _style_ax(ax, title: str):
    """Apply dark-theme styling to a matplotlib Axes."""
    ax.set_facecolor(_BG)
    ax.set_title(title, color=_FG, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)


def _dark_fig(w=10, h=3) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────── #


def plot_waveform(y: np.ndarray, sr: int) -> plt.Figure:
    """Raw waveform with a soft amplitude envelope overlay."""
    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))

    # Compute RMS envelope (frame-level energy)
    frame_len = 512
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop, n_fft=frame_len
    )

    fig, ax = _dark_fig(10, 2.5)
    ax.plot(times, y, color=_ACCENT, linewidth=0.4, alpha=0.7)
    ax.fill_between(rms_times, -rms, rms, color=_ACCENT, alpha=0.15)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    _style_ax(ax, "Waveform Audio")
    plt.tight_layout(pad=0.5)
    return fig


def plot_cqt(y: np.ndarray, sr: int, label: str = "") -> plt.Figure:
    """
    Constant-Q Transform spectrogram — the primary acoustic evidence plot.

    The upper CQT bins are kept below the Nyquist limit so that the wavelet
    bandwidth does not exceed the valid frequency range of the audio.
    """
    fmin = librosa.note_to_hz("C2")
    bins_per_octave = 12

    # For 16 kHz audio, 84 bins are too close to the 8 kHz Nyquist limit.
    # Using 83 bins leaves enough room for the bandwidth of the highest wavelet.
    n_bins = 83

    C = librosa.cqt(
        y,
        sr=sr,
        hop_length=256,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=0.0,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    fig, ax = _dark_fig(10, 4)
    img = librosa.display.specshow(
        C_db,
        sr=sr,
        x_axis="time",
        y_axis="cqt_note",
        hop_length=256,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        ax=ax,
        cmap="magma",
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)
    cbar.set_label("Amplitudo (dB)", color=_FG, fontsize=9)

    title = "Constant-Q Transform (CQT) Spectrogram"
    if label:
        title += f"  —  Prediksi: {label}"
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Frekuensi (CQT)")
    _style_ax(ax, title)
    plt.tight_layout(pad=0.5)
    return fig

def plot_spectral_profile(y: np.ndarray, sr: int) -> plt.Figure:
    """
    Mean spectral energy profile.

    The 6–8 kHz region is highlighted as an inspection zone only.
    This plot does not independently determine whether the audio is deepfake.
    """
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    mean_db = S_db.mean(axis=1)

    fig, ax = _dark_fig(10, 3)
    ax.plot(freqs / 1000, mean_db, color=_ACCENT, linewidth=1.2)

    # Highlight only as an inspection zone, not as proof of deepfake.
    ax.axvspan(
        6,
        8,
        color=_DANGER,
        alpha=0.12,
        label="Area inspeksi frekuensi tinggi (6–8 kHz)",
    )

    # Keep the annotation inside the visible plotting area.
    idx_7khz = np.searchsorted(freqs, 7000)
    idx_7khz = min(idx_7khz, len(mean_db) - 1)

    ax.annotate(
        "Area inspeksi artefak spektral\\n(bukan bukti otomatis deepfake)",
        xy=(7, mean_db[idx_7khz]),
        xytext=(4.55, mean_db.max() - 5),
        color=_DANGER,
        fontsize=8,
        arrowprops=dict(
            arrowstyle="->",
            color=_DANGER,
            lw=1,
        ),
    )

    ax.set_xlabel("Frekuensi (kHz)")
    ax.set_ylabel("Energi rata-rata (dB)")
    ax.set_xlim(0, sr / 2000)
    ax.legend(facecolor=_GRID, labelcolor=_FG, fontsize=8, framealpha=0.7)
    _style_ax(ax, "Profil Energi Spektral")
    plt.tight_layout(pad=0.5)
    return fig

# ═══════════════════════════════════════════════════════════════════════════ #
#  XAI — Attention weight extraction via forward hooks
# ═══════════════════════════════════════════════════════════════════════════ #


def _preprocess_for_hooks(audio_path: str) -> torch.Tensor:
    """
    Reproduce the exact same preprocessing pipeline as train.py:
    load → mono → resample 16 kHz → pre-emphasis → pad/crop to 64 600 samples
    Returns tensor [1, T] on CPU.
    """
    audio_np, file_sr = sf.read(audio_path)
    audio = torch.from_numpy(audio_np.astype(np.float32))

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.mean(dim=-1, keepdim=True).T

    if file_sr != SR:
        audio = torchaudio.functional.resample(audio, file_sr, SR)

    audio = torchaudio.functional.preemphasis(audio)  # α = 0.97 (default)

    # pad or crop — same logic as _pad_or_crop in train.py
    T = audio.shape[1]
    if T < MAX_LEN:
        audio = torch.nn.functional.pad(audio, (0, MAX_LEN - T))
    else:
        audio = audio[:, :MAX_LEN]

    return audio  # [1, T]


def extract_attention_weights(model: torch.nn.Module, audio_path: str) -> dict:
    """
    Attach forward hooks to KAN-HS-GAL / attention layers and run one
    forward pass to capture their outputs.

    The AASIST3 architecture uses heterogeneous graph attention layers (HS-GAL)
    with KAN replacing the standard linear attention.  If a layer returns a
    tuple (features, attn_weights), the second element is captured; otherwise
    the raw output tensor is stored.

    Returns a dict  {layer_name: tensor}  — empty if no matching layers found.
    """
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    keywords = ("gal", "attn", "attention", "hs_gal", "hsgal", "graphpool", "kan")

    for name, module in model.named_modules():
        lower = name.lower()
        if any(k in lower for k in keywords):

            def _make_hook(layer_name: str):
                def _hook(mod, inp, out):
                    try:
                        if isinstance(out, (tuple, list)) and len(out) > 1:
                            t = out[1]
                        else:
                            t = out[0] if isinstance(out, (tuple, list)) else out
                        if isinstance(t, torch.Tensor):
                            captured[layer_name] = t.detach().cpu()
                    except Exception:
                        pass

                return _hook

            hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        audio = _preprocess_for_hooks(audio_path)
        device = next(model.parameters()).device
        audio = audio.to(device)

        model.eval()
        with torch.no_grad():
            _ = model(audio)
    except Exception as exc:
        st.warning(f"Attention extraction forward pass failed: {exc}")
    finally:
        for h in hooks:
            h.remove()

    return captured


# ═══════════════════════════════════════════════════════════════════════════ #
#  XAI — Attention heatmap
# ═══════════════════════════════════════════════════════════════════════════ #


def plot_attention_heatmap(captured: dict, y: np.ndarray, sr: int) -> plt.Figure | None:
    """
    Overlay extracted attention scores on the waveform.
    Colour (cool → warm) encodes attention magnitude; red/yellow = most
    suspicious segments.

    Picks the first layer whose tensor can be squeezed to 1-D.
    """
    if not captured:
        return None

    attn_vec = None
    chosen_layer = ""

    for name, tensor in captured.items():
        flat = tensor.float().numpy().flatten()
        if len(flat) >= 4:
            attn_vec = flat
            chosen_layer = name
            break

    if attn_vec is None:
        return None

    # Normalise 0-1
    lo, hi = attn_vec.min(), attn_vec.max()
    attn_norm = (attn_vec - lo) / (hi - lo + 1e-8)

    # Resample attention vector to audio length
    x_src = np.linspace(0, 1, len(attn_norm))
    x_dst = np.linspace(0, 1, len(y))
    attn_audio = np.interp(x_dst, x_src, attn_norm)

    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))

    fig, ax = _dark_fig(10, 3)

    # Waveform in neutral colour
    ax.plot(times, y, color="#555577", linewidth=0.3, zorder=1)

    # Scatter overlay coloured by attention score
    sc = ax.scatter(
        times,
        y,
        c=attn_audio,
        cmap="YlOrRd",
        s=0.15,
        alpha=0.85,
        linewidths=0,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Attention Score", color=_FG, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=_FG, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_FG)

    ax.set_xlim(0, duration)
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    short_name = chosen_layer.split(".")[-1] if "." in chosen_layer else chosen_layer
    _style_ax(
        ax,
        f"Attention Heatmap  (layer: {short_name})\n"
        "Merah/kuning = segmen paling diperhatikan model saat deteksi",
    )
    plt.tight_layout(pad=0.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════ #
#  Streamlit app
# ═══════════════════════════════════════════════════════════════════════════ #

st.set_page_config(
    page_title="VCCC - Audio Deepfake Detection",
    page_icon="🎙️",
    layout="centered",
)

st.title("VCCC — Audio Deepfake Detection")
st.write(
    "Aplikasi deteksi audio asli atau deepfake dengan pilihan checkpoint model."
)

checkpoint_paths = discover_checkpoints()

if not checkpoint_paths:
    st.error(
        "Checkpoint model tidak ditemukan. Tambahkan file `.pt`, `.pth`, atau `.ckpt` "
        f"ke folder `{CHECKPOINT_DIR}`."
    )
    st.stop()

selected_checkpoint = st.selectbox(
    "Pilih model / checkpoint",
    options=checkpoint_paths,
    format_func=format_model_name,
    help=(
        "Daftar ini dibaca otomatis dari folder checkpoints. "
        "Tambahkan file checkpoint baru agar muncul sebagai pilihan."
    ),
)

selected_model_name = format_model_name(selected_checkpoint)
st.caption(f"Model aktif: `{selected_model_name}`")

if len(checkpoint_paths) == 1:
    st.info(
        "Saat ini baru ditemukan 1 checkpoint. Tambahkan checkpoint lain ke folder "
        f"`{CHECKPOINT_DIR}` untuk menampilkan pilihan model tambahan."
    )

uploaded_file = st.file_uploader(
    "Upload file audio",
    type=["wav", "flac", "mp3"],
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    st.write("**Informasi File**")
    col_a, col_b = st.columns(2)
    col_a.write(f"Nama file: `{uploaded_file.name}`")
    col_b.write(f"Ukuran file: `{uploaded_file.size / 1024:.2f} KB`")

    if st.button("Deteksi Audio"):
        suffix = os.path.splitext(uploaded_file.name)[1]
        uploaded_file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # ── 1. Detection ──────────────────────────────────────────────
            with st.spinner("Memuat model dan memproses audio..."):
                model = get_cached_model(selected_checkpoint)
                result = predict_audio(model, tmp_path)

            st.subheader("Hasil Deteksi")
            st.write(f"Model yang digunakan: **{selected_model_name}**")

            if result["class_id"] == 0:
                st.error(f"Label Prediksi: {result['label']}")
                st.warning(
                    "⚠️ Audio terindikasi sebagai hasil manipulasi atau deepfake."
                )
            else:
                st.success(f"Label Prediksi: {result['label']}")
                st.info("✅ Audio terindikasi sebagai audio asli (bona fide).")

            st.write(f"Confidence: **{result['confidence']:.2f}%**")

            with st.expander("Detail Probabilitas"):
                st.write(result["probabilities"])

            # ── 2. Explainable AI ─────────────────────────────────────────
            st.divider()
            st.subheader("📊 Visualisasi Bukti Akustik (Explainable AI)")
            st.caption(
                "Visualisasi di bawah membantu melakukan inspeksi terhadap karakteristik audio. "
                "Waveform, CQT Spectrogram, dan profil spektral merupakan visualisasi pendukung, "
                "bukan bukti otomatis bahwa audio merupakan deepfake. Hasil utama tetap mengacu "
                "pada prediksi dan confidence model."
            )

            with st.spinner("Memproses visualisasi akustik..."):
                y_xai, sr_xai = load_audio_for_xai(tmp_path)

            # ── Tabs ──────────────────────────────────────────────────────
            tab_wave, tab_cqt, tab_spec, tab_attn = st.tabs(
                [
                    "🌊 Waveform",
                    "🎵 CQT Spectrogram",
                    "📈 Profil Spektral",
                    "🔍 Attention Map",
                ]
            )

            # Tab 1 — Waveform
            with tab_wave:
                fig = plot_waveform(y_xai, sr_xai)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Bentuk gelombang mentah dengan envelope energi (RMS) yang dioverlay. "
                    "Transisi yang tidak wajar pada audio deepfake terkadang terlihat sebagai "
                    "lonjakan amplitudo yang tidak konsisten dengan konteks fonetis."
                )

            # Tab 2 — CQT Spectrogram
            with tab_cqt:
                fig = plot_cqt(y_xai, sr_xai, label=result["label"])
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "**Constant-Q Transform (CQT)** menggunakan resolusi frekuensi logaritmik yang "
                    "lebih tajam dari FFT standar dalam mendeteksi anomali spektral halus. "
                    "Area terang di frekuensi tinggi yang membentuk pola berulang (checkerboard) "
                    "mengindikasikan artefak periodisasi dari proses dekonvolusi AI."
                )

            # Tab 3 — Spectral Profile
            with tab_spec:
                fig = plot_spectral_profile(y_xai, sr_xai)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    "Profil energi rata-rata per frekuensi. Area merah 6–8 kHz adalah "
                    "**area inspeksi frekuensi tinggi**, bukan bukti otomatis bahwa audio "
                    "merupakan deepfake. Interpretasi utama tetap mengacu pada hasil prediksi "
                    "dan confidence model."
                )

            # Tab 4 — Attention Heatmap
            with tab_attn:
                with st.spinner(
                    "Mengekstrak attention weights dari lapisan KAN-HS-GAL..."
                ):
                    captured = extract_attention_weights(model, tmp_path)

                if captured:
                    fig = plot_attention_heatmap(captured, y_xai, sr_xai)

                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            "Heatmap ini bersifat **eksperimental**. Forward hook menangkap tensor "
                            "dari lapisan terkait attention/KAN-HS-GAL, lalu memproyeksikannya ke "
                            "domain waktu. Gunakan visualisasi ini sebagai pendukung interpretasi, "
                            "bukan sebagai bukti kausal tunggal."
                        )

                        with st.expander("Detail Layer yang Diekstrak"):
                            for lname, t in captured.items():
                                st.write(f"• `{lname}` — shape: `{tuple(t.shape)}`")
                    else:
                        st.info(
                            "Attention weights ditemukan namun bentuk tensor-nya tidak dapat "
                            "diproyeksikan ke domain waktu untuk visualisasi heatmap."
                        )
                else:
                    st.info(
                        "Attention Map tidak tersedia untuk checkpoint ini. "
                        "Lapisan KAN-HS-GAL mungkin tidak mengekspos bobot atensi "
                        "sebagai output terpisah pada arsitektur ini."
                    )
                    st.caption(
                        "💡 Untuk mengaktifkan Attention Map, modifikasi lapisan KAN-GAL agar "
                        "mengembalikan attention weights sebagai elemen kedua dari output tuple: "
                        "`return features, attn_weights`."
                    )

        except Exception as e:
            st.error("Terjadi error saat proses deteksi.")
            st.exception(e)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
