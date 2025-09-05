
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True nếu nhận GPU

import sys
sys.path.append("/content/vienam-TTS")  # Thêm thư mục /content vào đường dẫn tìm kiếm module

import torch
import json
import unicodedata
import regex
import numpy as np
import gradio as gr
from types import SimpleNamespace

# Giả định bạn đã upload các file mô hình:
# config.json, duration_model.pth, gen_630k.pth, phone_set.json

# Cấu hình
config_file = "config.json"
duration_model_path = "duration_model.pth"
lightspeed_model_path = "gen_630k.pth"
phone_set_file = "phone_set.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load cấu hình
with open(config_file, "rb") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

# Load phone set
with open(phone_set_file, "r") as f:
    phone_set = json.load(f)

sil_idx = phone_set.index("sil") if "sil" in phone_set else 0

# Hàm đọc số thành chữ
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

def read_number(num: str) -> str:
    if num.isdigit():
        if len(num) == 1:
            return digits[int(num)]
        elif len(num) == 2:
            n = int(num)
            end = digits[n % 10]
            if n < 20:
                return "mười " + end
            else:
                if n % 10 == 1:
                    end = "mốt"
                return digits[n // 10] + " mươi " + end
    return num

# Chuyển text sang danh sách index phoneme
def text_to_phone_idx(text):
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    words = text.split()
    words = [read_number(w) for w in words]
    tokens = []
    for c in " ".join(words):
        if c in phone_set:
            tokens.append(phone_set.index(c))
        elif c == " ":
            tokens.append(0)
    tokens = [sil_idx, 0] + tokens + [0, sil_idx]
    return tokens

# Hàm TTS
def text_to_speech(duration_net, generator, text):
    phone_idx = text_to_phone_idx(text)
    batch = {
        "phone_idx": np.array([phone_idx]),
        "phone_length": np.array([len(phone_idx)]),
    }

    phone_length = torch.from_numpy(batch["phone_length"]).long().to(device)
    phone_idx = torch.from_numpy(batch["phone_idx"]).long().to(device)

    with torch.inference_mode():
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000
        phone_duration = torch.where(phone_idx == sil_idx, torch.clamp_min(phone_duration, 200), phone_duration)
        phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

        end_time = torch.cumsum(phone_duration, dim=-1)
        start_time = end_time - phone_duration
        start_frame = start_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
        end_frame = end_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
        spec_length = end_frame.max(dim=-1).values
        pos = torch.arange(0, spec_length.item(), device=device)
        attn = torch.logical_and(
            pos[None, :, None] >= start_frame[:, None, :],
            pos[None, :, None] < end_frame[:, None, :],
        ).float()

        y_hat = generator.infer(phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.0)[0]
    wave = y_hat[0, 0].data.cpu().numpy()
    return (wave * (2**15)).astype(np.int16)

# Load mô hình một lần
def load_models():
    from models import DurationNet, SynthesizerTrn  # cần upload models.py từ repo

    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(duration_model_path, map_location=device))
    duration_net = duration_net.eval()

    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(lightspeed_model_path, map_location=device)
    params = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["net_g"].items()}
    generator.load_state_dict(params, strict=False)
    generator = generator.eval()
    return duration_net, generator

duration_net, generator = load_models()

# Hàm gọi từ Gradio
def speak(text):
    paragraphs = text.split("\n")
    clips = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            clips.append(text_to_speech(duration_net, generator, paragraph))
    if clips:
        y = np.concatenate(clips)
    else:
        y = np.zeros(16000, dtype=np.int16)
    return hps.data.sampling_rate, y

# Giao diện Gradio
gr.Interface(
    fn=speak,
    inputs="text",
    outputs="audio",
    title="LightSpeed: Vietnamese Female Voice TTS",
    description="Vietnamese Female Voice TTS",
    examples=[
        "Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau.",
        "Đoạn trường tân thanh, thường được biết đến với cái tên đơn giản là Truyện Kiều, là một truyện thơ của đại thi hào Nguyễn Du",
    ],
).launch(debug=True)
