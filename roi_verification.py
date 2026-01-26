import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.ndimage import zoom

# ==========================================
# 1. 설정 및 하이퍼파라미터
# ==========================================
CONFIG = {
    'data_dir': './Respiratory_Sound_Database/audio_and_txt_files', 
    'diagnosis_path': './Respiratory_Sound_Database/patient_diagnosis.csv',
    'model_path': 'resnet18_icbhi_copd.pth',
    'sample_rate': 16000,
    'n_mels': 128,
    'duration': 5,
    # [중요] ROI 기준점 설정 (Mel Bin Index)
    # n_mels=128일 때, 0~8000Hz 표현.
    # 보통 폐기종(Wheeze, Crackle)은 200~400Hz 이상에서 나타남.
    # 대략 15~20번째 bin 이상을 고주파 병변 영역으로 가정 (실험적으로 조정 가능)
    'roi_threshold_bin': 20 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 필요한 클래스 및 함수 재정의
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
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 데이터 로딩 함수 (Main 코드에서 가져옴, 간소화)
def get_test_samples(data_dir, diagnosis_path):
    # 실제로는 Train/Test Split을 정확히 해야 하지만, 여기서는 전체 스캔 예시
    # (실제 연구에선 Test Set만 따로 분리된 리스트를 사용해야 함)
    diagnosis_df = pd.read_csv(diagnosis_path, names=['pid', 'diagnosis'], sep=None, engine='python')
    pid_to_label = {}
    for _, row in diagnosis_df.iterrows():
        if row['diagnosis'] == 'COPD': pid_to_label[row['pid']] = 1
        elif row['diagnosis'] == 'Healthy': pid_to_label[row['pid']] = 0
            
    samples = []
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    # 빠른 테스트를 위해 100개만 샘플링 (전체는 시간이 걸림)
    for wav_file in audio_files[:100]: 
        pid = int(wav_file.split('_')[0])
        if pid not in pid_to_label: continue
        samples.append({
            'path': os.path.join(data_dir, wav_file),
            'label': pid_to_label[pid]
        })
    return samples

def preprocess(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != CONFIG['sample_rate']:
        resampler = T.Resample(sr, CONFIG['sample_rate'])
        waveform = resampler(waveform)
    
    # 5초 자르기 (단순화)
    target_len = CONFIG['sample_rate'] * CONFIG['duration']
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
        
    mel = T.MelSpectrogram(sample_rate=CONFIG['sample_rate'], n_mels=CONFIG['n_mels'])(waveform)
    mel = T.AmplitudeToDB()(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0).unsqueeze(0)

# ==========================================
# 3. 핵심: ROI Score 계산 함수
# ==========================================
def calculate_roi_score(heatmap, threshold_bin, original_height=128, original_width=157):
    """
    히트맵 에너지가 고주파(ROI) 영역에 얼마나 분포하는지 계산
    Formula: Sum(ROI) / Sum(Total)
    
    heatmap을 원본 mel-spectrogram 크기로 확대한 후 ROI 계산
    """
    # heatmap을 원본 크기로 확대 (4x5 -> 128x157)
    h, w = heatmap.shape
    zoom_factors = (original_height / h, original_width / w)
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    
    # ROI: threshold_bin 이상의 주파수 영역 (고주파 = 병변 영역)
    roi_energy = np.sum(heatmap_resized[threshold_bin:, :])
    total_energy = np.sum(heatmap_resized)
    
    if total_energy == 0: return 0.0
    
    score = roi_energy / total_energy
    return score

# ==========================================
# 4. 메인 실행 검증
# ==========================================
if __name__ == "__main__":
    print("--- Quantitative Reliability Verification ---")
    
    model = load_trained_model(CONFIG['model_path'])
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    samples = get_test_samples(CONFIG['data_dir'], CONFIG['diagnosis_path'])
    print(f"검증할 샘플 수: {len(samples)}개")
    
    roi_scores = []
    correct_copd_predictions = 0
    
    for item in tqdm(samples):
        # 1. 전처리
        input_tensor = preprocess(item['path']).to(device)
        label = item['label']
        
        # 2. 모델 예측
        if input_tensor.dim() == 5:
            input_tensor = input_tensor.squeeze(2)
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        
        # 3. 분석 조건: 
        # 실제 COPD(1)이고 모델도 COPD(1)라고 맞춘 경우에만 신뢰성 분석 진행
        if label == 1 and pred_idx == 1:
            heatmap, _ = grad_cam(input_tensor, class_idx=1)
            
            # ROI 점수 계산 (original_height=128은 n_mels와 동일)
            score = calculate_roi_score(heatmap, CONFIG['roi_threshold_bin'], CONFIG['n_mels'])
            roi_scores.append(score)
            correct_copd_predictions += 1
            
    # 결과 출력
    if roi_scores:
        avg_roi = np.mean(roi_scores)
        std_roi = np.std(roi_scores)
        print("\n[검증 결과]")
        print(f"분석된 True Positive(COPD) 샘플 수: {correct_copd_predictions}")
        print(f"평균 ROI Score: {avg_roi:.4f} (±{std_roi:.4f})")
        print("-" * 30)
        print("해석 가이드:")
        print(" - 점수가 1.0에 가까울수록: 모델이 고주파(병변) 영역만 보고 판단함.")
        print(" - 점수가 0.0에 가까울수록: 모델이 저주파(정상 호흡음)만 보고 판단함.")
        print(" - 보통 0.4 ~ 0.7 사이가 나오면 병변과 호흡음을 동시에 고려한다고 해석 가능.")
    else:
        print("COPD라고 정확히 예측한 샘플이 없어서 점수를 계산할 수 없습니다.")