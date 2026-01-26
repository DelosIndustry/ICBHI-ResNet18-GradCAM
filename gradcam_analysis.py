import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # cv2 대신 사용
import os

# ==========================================
# 1. 설정 (학습 코드와 동일하게 유지)
# ==========================================
CONFIG = {
    'sample_rate': 16000,
    'n_mels': 128,
    'duration': 5,
    'model_path': 'resnet18_icbhi_copd.pth',  # 학습된 모델 파일
    # 테스트하고 싶은 wav 파일 경로를 여기에 넣으세요
    'test_file': './Respiratory_Sound_Database/audio_and_txt_files/120_1b1_Ar_sc_Meditron.wav' 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 모델 로드 함수 (구조 재정의 필요)
# ==========================================
def load_trained_model(path):
    model = models.resnet18(pretrained=False) # 구조만 가져옴
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # 저장된 가중치 로드
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print("모델 가중치 로드 성공!")
    except FileNotFoundError:
        print("모델 파일이 없습니다. 먼저 학습(main.py)을 실행하세요.")
        exit()
        
    model.to(device)
    model.eval() # 평가 모드 (Dropout 등 비활성화)
    return model

# ==========================================
# 3. Grad-CAM 클래스 (핵심 로직)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook 등록 (중간 데이터를 낚아채는 역할)
        # 1. Forward Pass에서 특징맵(Activation) 저장
        self.target_layer.register_forward_hook(self.save_activation)
        # 2. Backward Pass에서 기울기(Gradient) 저장
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]이 해당 레이어의 기울기
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)
        
        if class_idx is None:
            # 별도 지정 없으면 가장 확률 높은 클래스 선택
            class_idx = torch.argmax(output, dim=1).item()
        
        # 2. Backward Pass (해당 클래스 스코어에 대해서만 미분)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # 3. Grad-CAM 계산
        # (1) Gradients의 Global Average Pooling (채널별 중요도 가중치)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # (2) 가중치와 Activation 곱하기
        activation = self.activations[0] # [C, H, W]
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        # (3) 채널 축으로 합치기 (Heatmap 생성)
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        
        # (4) ReLU 적용 (양의 영향력만 남김)
        heatmap = np.maximum(heatmap, 0)
        
        # (5) 0~1 정규화
        heatmap /= (np.max(heatmap) + 1e-7)
        
        return heatmap, output

# ==========================================
# 4. 오디오 전처리 (하나만 처리)
# ==========================================
def preprocess_audio(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    
    # 리샘플링
    if sr != CONFIG['sample_rate']:
        resampler = T.Resample(sr, CONFIG['sample_rate'])
        waveform = resampler(waveform)
    
    # 5초만 자르기 (앞부분) - 테스트용
    target_len = CONFIG['sample_rate'] * CONFIG['duration']
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        pad_amt = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        
    # Mel-Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=CONFIG['sample_rate'], n_fft=1024, hop_length=512, n_mels=CONFIG['n_mels']
    )
    db_transform = T.AmplitudeToDB()
    
    mel_spec = mel_transform(waveform)
    mel_spec = db_transform(mel_spec)
    
    # 정규화
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
    
    # 배치 차원 추가 [1, 1, 128, Time]
    input_tensor = mel_spec.unsqueeze(0)  # [1, 1, 1, 128, 157]
    return input_tensor, mel_spec.numpy() # 시각화용 원본 반환

# ==========================================
# 5. 실행 및 시각화
# ==========================================
if __name__ == "__main__":
    # 모델 로드
    model = load_trained_model(CONFIG['model_path'])
    
    # Grad-CAM 설정 (ResNet18의 마지막 합성곱 층: layer4의 마지막 블록)
    # ResNet 구조: layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # 오디오 처리
    if not os.path.exists(CONFIG['test_file']):
        print(f"테스트 파일이 없습니다: {CONFIG['test_file']}")
        # 더미 데이터 생성 (테스트용) - shape 수정: [1, 1, 128, 157]
        input_tensor = torch.randn(1, 1, 128, 157).to(device)
        original_mel = input_tensor.squeeze(0).cpu().numpy()  # [1, 128, 157] 유지
    else:
        input_tensor, original_mel = preprocess_audio(CONFIG['test_file'])
        input_tensor = input_tensor.to(device)
    
    # Grad-CAM 생성 (Target: 1 (COPD) 라고 가정하고 중요 영역 확인)
    heatmap, output = grad_cam(input_tensor, class_idx=1)
    
    # 결과 해석
    probs = F.softmax(output, dim=1)
    pred_idx = torch.argmax(probs).item()
    pred_label = "COPD" if pred_idx == 1 else "Healthy"
    confidence = probs[0][pred_idx].item() * 100
    
    print(f"Prediction: {pred_label} ({confidence:.2f}%)")
    
    # 시각화 (Superimpose)
    plt.figure(figsize=(12, 5))
    
    # (1) 원본 멜 스펙트로그램
    plt.subplot(1, 2, 1)
    plt.imshow(original_mel[0], aspect='auto', origin='lower', cmap='magma')
    plt.title("Original Mel-Spectrogram")
    plt.ylabel("Frequency (Mel bins)")
    plt.xlabel("Time")
    
    # (2) Grad-CAM Overlay
    plt.subplot(1, 2, 2)
    # cv2.resize 대신 scipy.ndimage.zoom 사용
    zoom_factors = (original_mel.shape[1] / heatmap.shape[0], 
                    original_mel.shape[2] / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    
    plt.imshow(original_mel[0], aspect='auto', origin='lower', cmap='gray')
    plt.imshow(heatmap_resized, aspect='auto', origin='lower', alpha=0.5, cmap='jet')
    plt.title(f"Grad-CAM (Focusing on COPD Class)")
    plt.ylabel("Frequency (Mel bins)")
    plt.xlabel("Time")
    
    plt.tight_layout()
    
    save_path = "gradcam_result.png"
    plt.savefig(save_path, dpi=300)  
    print(f"시각화 결과가 {save_path}에 저장되었습니다.")
    