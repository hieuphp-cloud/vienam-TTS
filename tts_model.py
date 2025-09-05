# tts_model.py
import torch
import torchaudio
from torch.utils.data import Dataset
import json
from types import SimpleNamespace
from models import DurationNet, SynthesizerTrn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config + phone set
with open("config.json", "r") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
with open("phone_set.json", "r") as f:
    phone_set = json.load(f)
sil_idx = phone_set.index("sil") if "sil" in phone_set else 0

# Dataset đơn giản
class TTSFineTuneDataset(Dataset):
    def __init__(self, audio_paths, transcript_paths):
        self.audio_paths = audio_paths
        self.transcript_paths = transcript_paths

    def text_to_phone_idx(self, text):
        text = text.lower()
        tokens = []
        for c in text:
            if c in phone_set:
                tokens.append(phone_set.index(c))
            else:
                tokens.append(sil_idx)
        return [sil_idx, 0] + tokens + [0, sil_idx]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.audio_paths[idx])
        audio = audio.mean(dim=0)  # mono
        audio = audio.numpy()
        with open(self.transcript_paths[idx], "r", encoding="utf-8") as f:
            text = f.read().strip()
        phone_idx = self.text_to_phone_idx(text)
        return torch.LongTensor(phone_idx), torch.FloatTensor(audio)

# Load model gốc
def load_models(duration_path="duration_model.pth", gen_path="gen_630k.pth"):
    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(duration_path, map_location=device))
    duration_net.train()  # enable training

    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(gen_path, map_location=device)
    params = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["net_g"].items()}
    generator.load_state_dict(params, strict=False)
    generator.train()  # enable training
    return duration_net, generator

# Fine-tune đơn giản
def fine_tune(duration_net, generator, dataloader, epochs=3, lr=1e-4):
    optimizer = torch.optim.Adam(list(duration_net.parameters()) + list(generator.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
       for phone_idx, audio in dataloader:
    # nếu batch_size=1, DataLoader tạo thêm 1 chiều -> squeeze
        if phone_idx.dim() == 3:
            phone_idx = phone_idx.squeeze(0)   # [seq_len] -> [1, seq_len]
            audio = audio.squeeze(0)           # [1, T] -> [T]
        
        phone_idx = phone_idx.to(device)
        phone_length = torch.LongTensor([phone_idx.shape[1]]).to(device)
        audio = audio.to(device).float()

        # forward duration_net
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000

        # attention map
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
            wave = y_hat[0, 0]
            min_len = min(wave.shape[0], audio.shape[1])
            loss = criterion(wave[:min_len], audio[:, :min_len].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss={loss.item():.4f}")
    return duration_net, generator
