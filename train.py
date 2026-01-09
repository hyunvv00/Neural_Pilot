import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

LOG_FILE = 'dataset/labels/labels.csv'
IMAGE_BASE_DIR = 'dataset/'
SEQUENCE_LENGTH = 10  # RNN 시퀀스 길이
BATCH_SIZE = 8    
EPOCHS = 10
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'best_model.pth' # 🚨 모델 저장 경로 변경

# 🚨🚨🚨 이미지 전처리 설정 (크롭 없이 전체 이미지 사용) 🚨🚨🚨
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
TARGET_IMG_HEIGHT = ORIGINAL_HEIGHT
TARGET_IMG_WIDTH = ORIGINAL_WIDTH  
IMAGE_CHANNELS = 3
CROP_OFFSET_HEIGHT = 0 

# PyTorch 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# --- Custom Loss Function (Dual Header Conditional MSE Loss) ---
# (Turn: Angular Weight 4.30 적용, Straight: Angular Weight 1.0 적용)
# ==============================================================================

class DualHeaderConditionalLoss(nn.Module):
    # 🚨 데이터 비율에 기반하여 가중치를 4.30으로 설정
    def __init__(self, angular_weight=1.0): 
        super(DualHeaderConditionalLoss, self).__init__()
        self.angular_weight = angular_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target, turn_mode): 
        # pred shape: (B, 4) -> [w_t, v_t, w_s, v_s]
        # target shape: (B, 2) -> [w, v]
        # turn_mode shape: (B, 1) -> 1.0 (True) / 0.0 (False)

        is_turn = (turn_mode == 1.0).squeeze(1) 
        is_straight = (turn_mode == 0.0).squeeze(1) 

        loss_turn = torch.tensor(0.0, device=pred.device)
        loss_straight = torch.tensor(0.0, device=pred.device)
        
        # 2. 턴 모드일 때 (Index 0, 1 사용, 가중치 적용)
        if is_turn.any():
            pred_turn = pred[is_turn, 0:2] # [w_t, v_t]
            target_turn = target[is_turn]  # [w, v]
            
            squared_error_turn = self.mse(pred_turn, target_turn)
            
            # 가중치 텐서 [angular_weight, 1.0]
            weights_turn = torch.tensor([self.angular_weight, 1.0], device=pred.device) 
            weighted_error_turn = squared_error_turn * weights_turn
            
            loss_turn = torch.mean(weighted_error_turn)

        # 3. 직진 모드일 때 (Index 2, 3 사용, 일반 MSE Loss)
        if is_straight.any():
            pred_straight = pred[is_straight, 2:4] # [w_s, v_s]
            target_straight = target[is_straight]  # [w, v]
            
            squared_error_straight = self.mse(pred_straight, target_straight)
            
            # 가중치 텐서 [1.0, 1.0]
            weights_straight = torch.tensor([1.0, 1.0], device=pred.device) 
            weighted_error_straight = squared_error_straight * weights_straight
            
            loss_straight = torch.mean(weighted_error_straight)

        # 4. 최종 손실
        total_loss = (loss_turn * is_turn.sum().float() + loss_straight * is_straight.sum().float()) / target.size(0)
        
        return total_loss

# ==============================================================================
# --- 2. PyTorch Dataset 클래스 (Turn Mode 입력 유지) ---
# ==============================================================================

class SequenceDataset(Dataset):
    def __init__(self, data_df, base_dir, sequence_length):
        self.base_dir = base_dir
        self.sequence_length = sequence_length
        self.data = self._create_sequences(data_df)
        
        if 'turn_mode' not in data_df.columns:
            raise ValueError("CSV 파일에 'turn_mode' 컬럼이 필요합니다.")

    def _create_sequences(self, data_df):
        data_df = data_df.reset_index(drop=True)
        num_samples = len(data_df)
        sequences = []

        for i in range(num_samples - self.sequence_length + 1):
            sequence = data_df.iloc[i : i + self.sequence_length]
            
            img_paths = sequence['image_path'].tolist()
            labels = sequence[['angular_velocity_z', 'linear_velocity_x']].values.astype(np.float32)
            turn_mode_data = sequence['turn_mode'].values.astype(np.float32).reshape(-1, 1)
            
            sequences.append((img_paths, labels, turn_mode_data))
        return sequences

    def __len__(self):
        return len(self.data)

    def _preprocess_image(self, image_path_relative):
        image_path_absolute = os.path.join(self.base_dir, image_path_relative)
        
        img = cv2.imread(image_path_absolute)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path_absolute}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 크롭 및 리사이즈 로직
        img = cv2.resize(img, (TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        
        img = (img.astype(np.float32) / 255.0) - 0.5
        img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
        
        return img

    def __getitem__(self, idx):
        img_paths, labels, turn_mode_data = self.data[idx]
        
        image_sequence = [self._preprocess_image(path) for path in img_paths]
            
        image_sequence = np.array(image_sequence, dtype=np.float32)
        
        images_tensor = torch.from_numpy(image_sequence) # (T, C, H, W)
        labels_tensor = torch.from_numpy(labels).float() # (T, 2)
        turn_mode_tensor = torch.from_numpy(turn_mode_data).float() # (T, 1)
        
        return images_tensor, labels_tensor, turn_mode_tensor

# ==============================================================================
# --- 3. PyTorch 모델 정의 (Dual Header 출력) ---
# ==============================================================================
DUMMY_INPUT_SIZE = (1, IMAGE_CHANNELS, TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH)

class Dave2LSTMModel(nn.Module):
    def __init__(self):
        super(Dave2LSTMModel, self).__init__()

        self.cnn_base = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 24, kernel_size=5, stride=2, padding=2), nn.ELU(), 
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2), nn.ELU(),   
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2), nn.ELU(),   
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0), nn.ELU(),   
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ELU(),   
            nn.Flatten()
        )
        
        try:
            with torch.no_grad():
                dummy_output = self.cnn_base(torch.zeros(*DUMMY_INPUT_SIZE))
                self.feature_dim = dummy_output.size(1) 
        except Exception as e:
            # 480x640 이미지 기준 272384
            self.feature_dim = 272384
            print(f"CNN Feature Dimension Calculation failed. Using fallback: {self.feature_dim}")
            
        print(f"CNN Feature Dimension calculated: {self.feature_dim}")
            
        # LSTM 입력: CNN 특징 + Bool 플래그 (self.feature_dim + 1)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim + 1, 
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )
        
        # 🚨 최종 출력 Dense Layer: LSTM 출력 128 -> 4 (Dual Header)
        self.output_dense = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(128, 4) 
        )

    def forward(self, x_seq, turn_mode_seq):
        B, T, C, H, W = x_seq.size()
        
        # 1. CNN 특징 추출: (B*T, feature_dim)
        cnn_input = x_seq.view(B * T, C, H, W)
        cnn_features = self.cnn_base(cnn_input) 
        
        # 2. LSTM 입력 형태 복원 및 Bool 플래그 결합
        lstm_cnn_input = cnn_features.view(B, T, self.feature_dim)
        lstm_input = torch.cat([lstm_cnn_input, turn_mode_seq], dim=2)
        
        # 3. LSTM 처리
        lstm_output, _ = self.lstm(lstm_input) # (B, T, 128)
        
        # 4. Dense Layer 입력 준비: LSTM의 마지막 시점 출력만 사용
        last_lstm_output = lstm_output[:, -1, :] # (B, 128)
        
        # 5. 최종 출력: (B, 4) -> [w_t, v_t, w_s, v_s]
        final_output = self.output_dense(last_lstm_output)
        
        return final_output 

# ==============================================================================
# --- 4. 훈련 루프 ---
# ==============================================================================

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # 훈련 단계
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", unit="batch")
        for images, labels_seq, turn_mode_seq in train_bar:
            images = images.to(DEVICE)
            labels = labels_seq[:, -1, :].to(DEVICE) # (B, 2)
            turn_mode_data = turn_mode_seq.to(DEVICE) # (B, T, 1)
            
            optimizer.zero_grad()
            
            outputs = model(images, turn_mode_data) # outputs shape: (B, 4)
            
            # Loss 계산: 마지막 프레임의 라벨과 턴 모드 사용
            turn_mode_last = turn_mode_data[:, -1, :] # (B, 1)
            loss = criterion(outputs, labels, turn_mode_last) 
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        # 검증 단계
        avg_val_loss = evaluate_model(model, val_loader, criterion)
        
        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss (Dual Header): {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> 모델 저장: Validation Loss 개선 ({best_val_loss:.6f}).")

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels_seq, turn_mode_seq in loader:
            images = images.to(DEVICE)
            labels = labels_seq[:, -1, :].to(DEVICE) 
            turn_mode_data = turn_mode_seq.to(DEVICE)
            
            outputs = model(images, turn_mode_data) # outputs shape: (B, 4)
            
            turn_mode_last = turn_mode_data[:, -1, :]
            loss = criterion(outputs, labels, turn_mode_last) 
            
            total_loss += loss.item()
            
    return total_loss / len(loader)


# ==============================================================================
# --- 5. 메인 실행 블록 ---
# ==============================================================================

if __name__ == '__main__':
    print(f"Device being used: {DEVICE}")

    # --- 데이터 로드 및 준비 ---
    try:
        data_df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print(f"오류: '{LOG_FILE}' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        exit()
        
    # 데이터 분할
    train_df, validation_df = train_test_split(data_df, test_size=0.2, random_state=42)
    
    # Dataset 생성
    try:
        train_dataset = SequenceDataset(train_df, IMAGE_BASE_DIR, SEQUENCE_LENGTH)
        val_dataset = SequenceDataset(validation_df, IMAGE_BASE_DIR, SEQUENCE_LENGTH)
    except ValueError as e:
        print(f"데이터셋 오류: {e}")
        exit()
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"총 데이터 프레임 수: {len(data_df)}")
    print(f"훈련 시퀀스 수 (T={SEQUENCE_LENGTH}): {len(train_dataset)}")
    print(f"검증 시퀀스 수 (T={SEQUENCE_LENGTH}): {len(val_dataset)}")

    # --- 모델 생성 및 컴파일 ---
    model = Dave2LSTMModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 🚨 Dual Header Conditional Loss 사용 (Angular Weight = 4.30)
    criterion = DualHeaderConditionalLoss(angular_weight=4.30) 

    print(f"\nPyTorch DAVE2 + LSTM Dual Header 모델 훈련을 시작합니다 (T={SEQUENCE_LENGTH}, Weight=4.30)...")
    
    # --- 학습 ---
    try:
        train_model(model, train_loader, val_loader, optimizer, criterion, EPOCHS)
    except Exception as e:
        print(f"\n훈련 중 오류 발생: {e}")
        print("시퀀스 데이터셋 구성이나 메모리 부족(OOM) 문제일 수 있습니다. BATCH_SIZE를 확인하세요.")
        
    print(f"\n최적 모델이 '{MODEL_SAVE_PATH}'로 저장되었습니다.")
