import torch
import torch.nn as nn
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
    'data_dir': './Respiratory_Sound_Database/audio_and_txt_files',  # 데이터 경로 확인
    'diagnosis_path': './Respiratory_Sound_Database/patient_diagnosis.csv',
    'model_path': 'resnet18_icbhi_copd.pth',
    'sample_rate': 16000,
    'n_mels': 128,
    'duration': 5,
    # [수정됨] 40 -> 20 (약 300Hz 이상을 병변으로 간주)
    'roi_threshold_bin': 20 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 클래스 및 함수 정의
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
        return heatmap

def load_trained_model(path):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # 저장된 모델 불러오기 (DataParallel로 저장되었을 경우 대비)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        # 혹시 'module.' prefix가 있어서 에러나면 처리
        state_dict = torch.load(path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    model.to(device)
    model.eval()
    return model

def preprocess(wav_path):
    try:
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
            
        mel = T.MelSpectrogram(sample_rate=CONFIG['sample_rate'], n_mels=CONFIG['n_mels'])(waveform)
        mel = T.AmplitudeToDB()(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def get_test_samples(data_dir, diagnosis_path):
    # 실제 논문용: 전체 데이터를 다 가져와야 통계적 의미가 있음
    diagnosis_df = pd.read_csv(diagnosis_path, names=['pid', 'diagnosis'], sep=None, engine='python')
    pid_to_label = {}
    for _, row in diagnosis_df.iterrows():
        if row['diagnosis'] == 'COPD': pid_to_label[row['pid']] = 1
        elif row['diagnosis'] == 'Healthy': pid_to_label[row['pid']] = 0
            
    samples = []
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    # 시간 절약을 위해 100개만 할 수도 있지만, 논문용으론 전체 권장
    # 여기서는 전체로 설정
    for wav_file in audio_files: 
        pid = int(wav_file.split('_')[0])
        if pid not in pid_to_label: continue
        samples.append({
            'path': os.path.join(data_dir, wav_file),
            'label': pid_to_label[pid]
        })
    return samples

# ==========================================
# 3. 핵심: ROI Score 계산 (수정됨)
# ==========================================
def calculate_roi_score(heatmap, threshold_bin, input_width):
    # [수정] 원본 mel-spectrogram의 크기에 맞춰 동적으로 확대
    # Height는 n_mels(128)로 고정, Width는 입력에 따라 달라짐
    original_height = 128
    
    h, w = heatmap.shape
    zoom_factors = (original_height / h, input_width / w)
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    
    # ROI: threshold_bin 이상의 주파수 영역
    roi_energy = np.sum(heatmap_resized[threshold_bin:, :])
    total_energy = np.sum(heatmap_resized)
    
    if total_energy == 0: return 0.0
    return roi_energy / total_energy

# ==========================================
# 4. 메인 실행
# ==========================================
if __name__ == "__main__":
    print("--- 🧪 Quantitative Reliability Verification (ROI Analysis) ---")
    
    model = load_trained_model(CONFIG['model_path'])
    # ResNet18의 마지막 Convolution Layer (layer4의 마지막 블록)
    target_layer = model.layer4[-1] 
    grad_cam = GradCAM(model, target_layer)
    
    samples = get_test_samples(CONFIG['data_dir'], CONFIG['diagnosis_path'])
    print(f"전체 샘플 수: {len(samples)}개")
    print("분석 진행 중... (시간이 조금 걸릴 수 있습니다)")
    
    roi_scores = []
    analyzed_count = 0
    
    # 나중에 Best/Worst 케이스 확인을 위해 저장
    results_detail = []

    for item in tqdm(samples):
        input_tensor = preprocess(item['path'])
        if input_tensor is None: continue
        
        # 원본 Width 저장 (ROI 계산용)
        input_width = input_tensor.shape[3]
        input_tensor = input_tensor.to(device)
        label = item['label']
        
        # 모델 예측
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        
        # ★ 핵심 조건: True Positive (실제 COPD이고, 모델도 COPD라 한 경우)만 분석
        if label == 1 and pred_idx == 1:
            heatmap = grad_cam(input_tensor, class_idx=1)
            
            # ROI Score 계산
            score = calculate_roi_score(heatmap, CONFIG['roi_threshold_bin'], input_width)
            roi_scores.append(score)
            analyzed_count += 1
            
            results_detail.append({
                'filename': os.path.basename(item['path']),
                'score': score
            })

    # 결과 요약
    if roi_scores:
        avg_roi = np.mean(roi_scores)
        std_roi = np.std(roi_scores)
        
        print("\n" + "="*30)
        print(f"[검증 결과]")
        print(f"분석된 True Positive(COPD) 샘플 수: {analyzed_count}")
        print(f"평균 ROI Score: {avg_roi:.4f} (±{std_roi:.4f})")
        print("="*30)
        
        # Best Case & Worst Case 출력 (나중에 논문 그림용 파일 찾기 편하게)
        results_detail.sort(key=lambda x: x['score'], reverse=True)
        print("\n[Best Cases - 모델이 병변을 아주 잘 본 경우]")
        for res in results_detail[:3]:
            print(f"{res['filename']}: {res['score']:.4f}")
            
        print("\n[Worst Cases - 모델이 딴 곳(잡음)을 본 경우]")
        for res in results_detail[-3:]:
            print(f"{res['filename']}: {res['score']:.4f}")
            
        print("\n------------------------------")
        print("해석 가이드:")
        print(" - 점수가 1.0에 가까울수록: 모델이 고주파(병변) 영역만 보고 판단함.")
        print(" - 점수가 0.0에 가까울수록: 모델이 저주파(정상 호흡음)만 보고 판단함.")
        print(" - 보통 0.4 ~ 0.7 사이가 나오면 병변과 호흡음을 동시에 고려한다고 해석 가능.")
    else:
        print("분석할 True Positive 샘플이 없습니다.")