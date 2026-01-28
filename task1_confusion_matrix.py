import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import os
from tqdm import tqdm

# ==========================================
# 1. 설정 (기존 코드와 동일하게 유지)
# ==========================================
CONFIG = {
    'data_dir': './Respiratory_Sound_Database/audio_and_txt_files',    # 데이터 경로 확인 필수
    'diagnosis_path': './Respiratory_Sound_Database/patient_diagnosis.csv',
    'model_path': 'resnet18_icbhi_copd.pth',
    'sample_rate': 16000,
    'n_mels': 128,
    'duration': 5,
    'batch_size': 32
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 모델 및 전처리 함수 (재사용)
# ==========================================
def load_trained_model(path):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✅ 모델 로드 성공: {path}")
    except FileNotFoundError:
        print(f"❌ 모델 파일이 없습니다: {path}")
        print("먼저 main.py를 실행하여 모델을 학습시켜주세요.")
        exit()
        
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
        return mel.unsqueeze(0)  # [1, 1, 128, Time] - 4D 텐서
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def get_eval_data(data_dir, diagnosis_path):
    # (주의) 실제 논문용으로는 Test Set만 따로 분리된 리스트를 가져와야 합니다.
    # 여기서는 코드 작동 확인을 위해 전체 데이터를 스캔합니다.
    diagnosis_df = pd.read_csv(diagnosis_path, names=['pid', 'diagnosis'], sep=None, engine='python')
    pid_to_label = {}
    for _, row in diagnosis_df.iterrows():
        if row['diagnosis'] == 'COPD': pid_to_label[row['pid']] = 1
        elif row['diagnosis'] == 'Healthy': pid_to_label[row['pid']] = 0
            
    samples = []
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    # 시간 절약을 위해 200개만 샘플링 (전체 평가시에는 슬라이싱 제거: audio_files[:])
    for wav_file in audio_files: 
        pid = int(wav_file.split('_')[0])
        if pid not in pid_to_label: continue
        samples.append({
            'path': os.path.join(data_dir, wav_file),
            'label': pid_to_label[pid]
        })
    return samples

# ==========================================
# 3. 메인 실행: 평가 및 시각화
# ==========================================
if __name__ == "__main__":
    print("--- 📊 Confusion Matrix 생성 및 성능 평가 ---")
    
    # 1. 모델 및 데이터 로드
    model = load_trained_model(CONFIG['model_path'])
    samples = get_eval_data(CONFIG['data_dir'], CONFIG['diagnosis_path'])
    print(f"평가 대상 샘플 수: {len(samples)}개")
    
    y_true = []
    y_pred = []
    
    # 2. 예측 수행
    print("예측 진행 중...")
    for item in tqdm(samples):
        input_tensor = preprocess(item['path'])
        if input_tensor is None: continue
        
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            
        y_true.append(item['label'])
        y_pred.append(pred_idx)
        
    # 3. Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 4. 성능 지표 계산
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred) # 민감도 (Sensitivity)
    specificity = recall_score(y_true, y_pred, pos_label=0) # 특이도
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"✅ Accuracy (정확도): {acc*100:.2f}%")
    print(f"✅ Sensitivity (민감도/Recall - COPD 탐지율): {recall*100:.2f}%")
    print(f"✅ Specificity (특이도 - 정상 구분율): {specificity*100:.2f}%")
    print(f"✅ F1-Score: {f1:.4f}")
    print("="*40)
    
    # 5. 시각화 (Seaborn Heatmap)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    
    # 라벨 정의
    class_names = ['Healthy (0)', 'COPD (1)']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix: ResNet-18 (COPD Detection)', fontweight='bold', pad=20)
    
    # 결과 저장 및 출력
    save_path = 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n📄 오차 행렬 이미지가 저장되었습니다: {save_path}")
    # plt.show()  # 임계값 실험 후 마지막에 표시

    # ==========================================
    # 4. 임계값(Threshold) 조정 실험
    # ==========================================
    # 기본은 0.5지만, 불균형 데이터에선 이걸 움직여서 성능을 찾기도 함
    # 낮은 임계값으로 테스트
    thresholds = [0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
    
    print("\n--- 🧪 임계값 조정 실험 ---")
    
    # 모델 출력값(Logits)을 확률(Probability)로 변환
    # y_pred_probs는 COPD일 확률 (0.0 ~ 1.0)
    y_pred_probs = []
    y_true_all = []
    
    model.eval()
    with torch.no_grad():
        for item in tqdm(samples):
            input_tensor = preprocess(item['path'])
            if input_tensor is None: continue
            input_tensor = input_tensor.to(device)
            
            output = model(input_tensor)
            # Softmax를 거쳐 확률 계산
            probs = torch.nn.functional.softmax(output, dim=1)
            copd_prob = probs[0][1].item() # 1번 클래스(COPD)일 확률
            
            y_pred_probs.append(copd_prob)
            y_true_all.append(item['label'])

    # 임계값 별로 성능 다시 계산
    for th in thresholds:
        # 확률이 th보다 크면 1(COPD), 아니면 0(Healthy)
        y_pred_th = [1 if p > th else 0 for p in y_pred_probs]
        
        acc = accuracy_score(y_true_all, y_pred_th)
        spec = recall_score(y_true_all, y_pred_th, pos_label=0) # 특이도
        sens = recall_score(y_true_all, y_pred_th, pos_label=1) # 민감도
        
        print(f"Threshold {th:.2f} -> Acc: {acc*100:.1f}% | Spec(정상구분): {spec*100:.1f}% | Sens(환자탐지): {sens*100:.1f}%")
    
    plt.show()