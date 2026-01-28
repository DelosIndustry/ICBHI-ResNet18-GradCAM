import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import os
import shutil # 파일 복사용
from tqdm import tqdm

# ==========================================
# 1. 설정 (Threshold 0.48 확정!)
# ==========================================
CONFIG = {
    'data_dir': './Respiratory_Sound_Database/audio_and_txt_files',    
    'diagnosis_path': './Respiratory_Sound_Database/patient_diagnosis.csv',
    'model_path': 'resnet18_icbhi_copd.pth',
    'sample_rate': 16000,
    'n_mels': 128,
    'duration': 5,
    'final_threshold': 0.48,  # 황금 임계값 적용
    'save_dir': './result_images' # 결과 이미지 저장 폴더
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 결과 저장 폴더 생성
if not os.path.exists(CONFIG['save_dir']):
    os.makedirs(CONFIG['save_dir'])

# ==========================================
# 2. 모델 및 함수 (이전과 동일)
# ==========================================
def load_trained_model(path):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # DataParallel로 저장된 경우 처리
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
        return mel.unsqueeze(0).unsqueeze(0)
    except:
        return None

def get_eval_data(data_dir, diagnosis_path):
    diagnosis_df = pd.read_csv(diagnosis_path, names=['pid', 'diagnosis'], sep=None, engine='python')
    pid_to_label = {}
    for _, row in diagnosis_df.iterrows():
        if row['diagnosis'] == 'COPD': pid_to_label[row['pid']] = 1
        elif row['diagnosis'] == 'Healthy': pid_to_label[row['pid']] = 0
            
    samples = []
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    for wav_file in audio_files: 
        pid = int(wav_file.split('_')[0])
        if pid not in pid_to_label: continue
        samples.append({'path': os.path.join(data_dir, wav_file), 'label': pid_to_label[pid]})
    return samples

# ==========================================
# 3. 메인 실행
# ==========================================
if __name__ == "__main__":
    print(f"--- 🚀 최종 평가 (Threshold: {CONFIG['final_threshold']}) ---")
    
    model = load_trained_model(CONFIG['model_path'])
    samples = get_eval_data(CONFIG['data_dir'], CONFIG['diagnosis_path'])
    
    y_true = []
    y_pred_th = []
    
    # ROI 점수 등 나중에 분석할 데이터를 저장할 리스트
    analysis_results = []

    print("평가 및 분석 진행 중...")
    with torch.no_grad():
        for item in tqdm(samples):
            input_tensor = preprocess(item['path'])
            if input_tensor is None: continue
            input_tensor = input_tensor.to(device)
            
            if input_tensor.dim() == 5:
                input_tensor = input_tensor.squeeze(2)
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            copd_prob = probs[0][1].item()
            
            # 임계값 적용
            pred_label = 1 if copd_prob > CONFIG['final_threshold'] else 0
            
            y_true.append(item['label'])
            y_pred_th.append(pred_label)
            
            # 결과 저장 (파일명, 실제값, 예측확률, 예측값)
            analysis_results.append({
                'filename': os.path.basename(item['path']),
                'path': item['path'],
                'label': item['label'],
                'prob': copd_prob,
                'pred': pred_label
            })

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_th)
    acc = accuracy_score(y_true, y_pred_th)
    sens = recall_score(y_true, y_pred_th, pos_label=1)
    spec = recall_score(y_true, y_pred_th, pos_label=0)
    
    print("\n" + "="*40)
    print(f"✅ Accuracy: {acc*100:.2f}%")
    print(f"✅ Sensitivity (COPD): {sens*100:.2f}%")
    print(f"✅ Specificity (Healthy): {spec*100:.2f}%")
    print("="*40)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'COPD'], yticklabels=['Healthy', 'COPD'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f"Confusion Matrix (Th={CONFIG['final_threshold']})")
    plt.savefig(os.path.join(CONFIG['save_dir'], 'final_confusion_matrix.png'), dpi=300)
    print("📸 Confusion Matrix 저장 완료!")

    # 2. 대표 이미지 추출 (Task 3)
    # Best Case: 실제 COPD(1)인데 예측 확률이 매우 높은 상위 5개
    df = pd.DataFrame(analysis_results)
    copd_samples = df[df['label'] == 1]
    
    best_cases = copd_samples.nlargest(5, 'prob')
    worst_cases = copd_samples.nsmallest(5, 'prob') # 실제 COPD인데 확률이 낮은(틀렸거나 간당간당한) 경우
    
    print("\n📸 대표 이미지 저장 중...")
    
    # Best Cases 저장
    for _, row in best_cases.iterrows():
        src = row['path']
        dst = os.path.join(CONFIG['save_dir'], f"Best_Conf{row['prob']:.4f}_{row['filename']}")
        shutil.copy(src, dst)
        
    # Worst Cases 저장
    for _, row in worst_cases.iterrows():
        src = row['path']
        dst = os.path.join(CONFIG['save_dir'], f"Worst_Conf{row['prob']:.4f}_{row['filename']}")
        shutil.copy(src, dst)
        
    print(f"✅ Best/Worst 케이스 이미지들이 '{CONFIG['save_dir']}' 폴더에 저장되었습니다.")