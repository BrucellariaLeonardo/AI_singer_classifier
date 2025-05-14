import torchaudio
import torch


def _preacondicionar_audio(signal, native_sr, target_sr):
  #resample
  if( native_sr != target_sr):
    resampler = torchaudio.transforms.Resample(native_sr, target_sr)
    signal = resampler(signal)
  #conversion a mono
  if(signal.shape[0] > 1):
    signal = torch.mean(signal, dim=0)
  return signal, target_sr


  fPath = "/content/drive/MyDrive/Colab/02.mp3"
target_sr = 16000

signal, sr = torchaudio.load(fPath)
signal, sr = _preacondicionar_audio(signal, sr, target_sr)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                       n_fft = 1024,
                                                       hop_length = 512,
                                                       n_mels = 64)