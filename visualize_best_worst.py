import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.ndimage import zoom

# ==========================================
# 1. 설정
# ==========================================
CONFIG = {
    'model_path': 'resnet18_icbhi_copd.pth',
    'target_folder': './result_images',  # Best/Worst wav 파일이 있는 곳
    'sample_rate': 16000,
    'n_mels': 128,
    'duration': 5
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 모델 및 Grad-CAM 클래스 (재사용)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-7)
        return heatmap, output

def load_trained_model(path):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # DataParallel 처리
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k.replace('module.', '')] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def preprocess_for_vis(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != CONFIG['sample_rate']:
        resampler = T.Resample(sr, CONFIG['sample_rate'])
        waveform = resampler(waveform)
    
    target_len = CONFIG['sample_rate'] * CONFIG['duration']
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
        
    mel_transform = T.MelSpectrogram(sample_rate=CONFIG['sample_rate'], n_mels=CONFIG['n_mels'])
    mel_spec = mel_transform(waveform)
    mel_spec = T.AmplitudeToDB()(mel_spec)
    
    # 정규화 (시각화할 때 대비가 잘 보이도록)
    mel_numpy = mel_spec.squeeze().numpy()
    
    # 모델 입력용 텐서
    input_tensor = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor, mel_numpy

# ==========================================
# 3. 메인 실행: 폴더 내 모든 wav 파일 시각화
# ==========================================
if __name__ == "__main__":
    print(f"--- 📸 Grad-CAM 시각화 시작: {CONFIG['target_folder']} ---")
    
    model = load_trained_model(CONFIG['model_path'])
    target_layer = model.layer4[-1] # ResNet 마지막 레이어
    grad_cam = GradCAM(model, target_layer)
    
    # 폴더 내 wav 파일 찾기
    wav_files = [f for f in os.listdir(CONFIG['target_folder']) if f.endswith('.wav')]
    
    if not wav_files:
        print("저장된 wav 파일이 없습니다. final_evaluation.py를 먼저 실행하세요.")
        exit()
        
    for wav_file in wav_files:
        full_path = os.path.join(CONFIG['target_folder'], wav_file)
        print(f"Processing: {wav_file}...")
        
        input_tensor, original_mel = preprocess_for_vis(full_path)
        input_tensor = input_tensor.to(device)
        
        # 1. Grad-CAM 추출 (Target: COPD Class = 1)
        heatmap, output = grad_cam(input_tensor, class_idx=1)
        
        # 2. Heatmap 크기 맞추기 (Zoom 이용 - 네가 칭찬받은 방식!)
        h, w = heatmap.shape
        zoom_factors = (original_mel.shape[0] / h, original_mel.shape[1] / w)
        heatmap_resized = zoom(heatmap, zoom_factors, order=1)
        
        # 3. 시각화 및 저장
        plt.figure(figsize=(12, 5))
        
        # (왼쪽) 원본 멜 스펙트로그램
        plt.subplot(1, 2, 1)
        plt.imshow(original_mel, aspect='auto', origin='lower', cmap='magma')
        plt.title(f"Original Mel-Spec\n({wav_file})", fontsize=10)
        plt.ylabel("Frequency (Mel bins)")
        plt.xlabel("Time")
        plt.colorbar(format='%+2.0f dB')

        # (오른쪽) Grad-CAM Overlay
        plt.subplot(1, 2, 2)
        plt.imshow(original_mel, aspect='auto', origin='lower', cmap='gray') # 배경 흑백
        plt.imshow(heatmap_resized, aspect='auto', origin='lower', alpha=0.6, cmap='jet') # 히트맵 덮기
        
        # 예측 확률 정보 표시
        probs = torch.nn.functional.softmax(output, dim=1)
        copd_prob = probs[0][1].item()
        plt.title(f"Grad-CAM (COPD Prob: {copd_prob*100:.1f}%)", fontsize=12, fontweight='bold')
        plt.ylabel("Frequency (Mel bins)")
        plt.xlabel("Time")
        plt.colorbar(label='Attention Score')

        plt.tight_layout()
        
        # 이미지 저장
        save_name = wav_file.replace('.wav', '_GradCAM.png')
        save_path = os.path.join(CONFIG['target_folder'], save_name)
        plt.savefig(save_path, dpi=300)
        plt.close() # 메모리 해제
        
    print("\n✅ 모든 시각화 이미지가 저장되었습니다!")
    print(f"확인 경로: {CONFIG['target_folder']}")