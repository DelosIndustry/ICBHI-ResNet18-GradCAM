import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 진행상황 바 표시용

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    'data_dir': './Respiratory_Sound_Database/audio_and_txt_files',
    'diagnosis_path': './Respiratory_Sound_Database/patient_diagnosis.csv',
    'sample_rate': 16000,                  # 타겟 샘플링 레이트
    'duration': 5,                         # 호흡 주기 고정 길이 (초)
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 1e-4,
    'n_mels': 128
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")

# ==========================================
# 2. 데이터 파싱 및 리스트 생성 함수
# ==========================================
def get_data_list(data_dir, diagnosis_path):
    print("데이터 리스트 생성 중...")
    try:
        diagnosis_df = pd.read_csv(diagnosis_path, names=['pid', 'diagnosis'], sep=None, engine='python')
    except:
        print("CSV 로드 실패. 파일 형식이나 경로를 확인해주세요.")
        return []

    # 2. 환자 ID -> 진단 매핑 딕셔너리 생성
    # 연구 목표: COPD(1) vs Healthy(0)
    pid_to_label = {}
    for index, row in diagnosis_df.iterrows():
        diag = row['diagnosis']
        if diag == 'COPD':
            pid_to_label[row['pid']] = 1
        elif diag == 'Healthy':
            pid_to_label[row['pid']] = 0
        # Asthma, LRTI 등 다른 질병은 이번 연구에서 제외 (원한다면 수정 가능)

    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    full_data_list = [] # (wav_path, start_time, end_time, label)

    for wav_file in tqdm(audio_files):
        # 파일명에서 환자 ID 추출 (예: 101_1b1... -> 101)
        pid = int(wav_file.split('_')[0])
        
        # 우리가 원하는 타겟(COPD, Healthy)이 아니면 스킵
        if pid not in pid_to_label:
            continue
            
        label = pid_to_label[pid]
        txt_file = wav_file.replace('.wav', '.txt')
        txt_path = os.path.join(data_dir, txt_file)
        wav_path = os.path.join(data_dir, wav_file)
        
        # 텍스트 파일(시간 정보) 읽기
        # 포맷: Start  End  Crackles  Wheezes
        try:
            # annotation 파일 로드
            ann = pd.read_csv(txt_path, sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
            
            for _, row in ann.iterrows():
                # 개별 호흡 주기의 시작/끝 시간 저장
                full_data_list.append({
                    'path': wav_path,
                    'start': row['start'],
                    'end': row['end'],
                    'label': label
                })
        except Exception as e:
            print(f"Error parsing {txt_file}: {e}")
            continue

    print(f"총 추출된 호흡 주기 샘플 수: {len(full_data_list)}")
    return full_data_list

# ==========================================
# 3. 데이터셋 클래스 (Dataset)
# ==========================================
class ICBHICycleDataset(Dataset):
    def __init__(self, data_list, config):
        self.data_list = data_list
        self.sr = config['sample_rate']
        self.duration = config['duration']
        self.target_len = self.sr * self.duration
        
        # 멜 스펙트로그램 변환기
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr, n_fft=1024, hop_length=512, n_mels=config['n_mels']
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 오디오 로드 (전체 로드 후 자르기)
        # (최적화를 위해선 torchaudio.load의 frame_offset을 쓰는 게 좋지만, 
        # mp3/wav 호환성 및 단순함을 위해 로드 후 슬라이싱 방식을 채택)
        waveform, origin_sr = torchaudio.load(item['path'])
        
        # 2. 리샘플링
        if origin_sr != self.sr:
            resampler = T.Resample(origin_sr, self.sr)
            waveform = resampler(waveform)
            
        # 3. 특정 구간(Cycle) 자르기
        start_sample = int(item['start'] * self.sr)
        end_sample = int(item['end'] * self.sr)
        
        # 범위 체크
        if start_sample >= waveform.shape[1]: start_sample = 0
        if end_sample > waveform.shape[1]: end_sample = waveform.shape[1]
        
        cycle_wave = waveform[:, start_sample:end_sample]
        
        # 4. 길이 고정 (Padding or Truncation)
        curr_len = cycle_wave.shape[1]
        if curr_len > self.target_len:
            cycle_wave = cycle_wave[:, :self.target_len]
        else:
            pad_amt = self.target_len - curr_len
            cycle_wave = torch.nn.functional.pad(cycle_wave, (0, pad_amt))
            
        # 5. 멜 스펙트로그램 변환
        mel_spec = self.mel_transform(cycle_wave)
        mel_spec = self.db_transform(mel_spec) # [1, n_mels, time]
        
        # 정규화 (선택 사항이나 학습 안정성을 위해 권장)
        mean = mel_spec.mean()
        std = mel_spec.std()
        mel_spec = (mel_spec - mean) / (std + 1e-6)

        return mel_spec, torch.tensor(item['label'], dtype=torch.long)

# ==========================================
# 4. 모델 정의 (ResNet-18)
# ==========================================
def get_model():
    model = models.resnet18(pretrained=True)
    # 입력 채널 수정 (3 -> 1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 출력 노드 수정 (Binary: 2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

# ==========================================
# 5. 메인 실행 루틴
# ==========================================
if __name__ == '__main__':
    # 1. 데이터 리스트 만들기
    full_data = get_data_list(CONFIG['data_dir'], CONFIG['diagnosis_path'])
    
    if len(full_data) == 0:
        print("데이터를 찾을 수 없습니다. 경로를 확인하세요.")
        exit()

    # 2. Train / Val 분리
    # 환자 기준으로 나누는 게 가장 좋지만, 우선 간단히 셔플 후 분할 (8:2)
    train_data, val_data = train_test_split(full_data, test_size=0.2, random_state=42, stratify=[x['label'] for x in full_data])

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # 3. DataLoader 생성
    train_ds = ICBHICycleDataset(train_data, CONFIG)
    val_ds = ICBHICycleDataset(val_data, CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0) # 윈도우라면 num_workers=0 권장
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 4. 불균형 가중치 계산 (수정된 부분)
    labels = [x['label'] for x in train_data]
    num_healthy = labels.count(0)
    num_copd = labels.count(1)
    
    print(f"클래스 분포 - Healthy: {num_healthy}, COPD: {num_copd}")
    
    total = num_healthy + num_copd
    
    # [수정됨] Healthy(0)에 강제 가중치 10배 부여!
    # 이유: Threshold 0.9에서도 특이도가 5.7%밖에 안 나와서, 모델이 정상을 더 중요하게 보도록 강제함.
    weight_0 = (total / (2 * num_healthy)) * 2.0 
    weight_1 = (total / (2 * num_copd)) * 1.0  # COPD는 그대로
    
    print(f"적용된 가중치 - Healthy: {weight_0:.4f}, COPD: {weight_1:.4f}")
    
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    
    # 5. 모델, Loss, Optimizer
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # 가중치 적용된 Loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 6. 학습 루프
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {100.*correct/total:.2f}%")

    print("\n학습 완료! 모델을 저장합니다.")
    torch.save(model.state_dict(), "resnet18_icbhi_copd.pth")