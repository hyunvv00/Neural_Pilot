import pandas as pd
import numpy as np
import cv2
import random
import os
import sys
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 320, 240, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS) 
BATCH_SIZE = 30
EPOCHS = 50
LEARNING_RATE = 2e-5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = SCRIPT_DIR
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets')
LOG_FILE_PATH = os.path.join(DATA_PATH, labels, 'label.csv')
FINAL_MODEL_SAVE_PATH = 'final_model.pth' 

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️ 사용 가능한 GPU를 찾을 수 없습니다. CPU로 훈련을 진행합니다...")
    return device

def preprocess_image(img):
    img_resized = cv2.resize(img, (320, 240))
    img_roi = img_resized[120:240, 0:160]
    img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    img_final = cv2.resize(img_roi, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return img_final

def augment_image(img, steering_angle):
    augmented_img = img.copy()
    augmented_angle = steering_angle
    
    if random.random() > 0.5:
        augmented_img = cv2.flip(augmented_img, 1)
        augmented_angle *= -1.0
    
    if random.random() > 0.7:
        brightness_factor = random.uniform(0.7, 1.3)
        augmented_img = cv2.convertScaleAbs(augmented_img, alpha=brightness_factor, beta=0)
    
    if random.random() > 0.8:
        h, w = augmented_img.shape[:2]
        mask = np.ones_like(augmented_img, dtype=np.float32)
        shadow_height = random.randint(50, 150)
        shadow_width = random.randint(100, 300)
        shadow_x = random.randint(0, w - shadow_width)
        shadow_y = random.randint(0, h - shadow_height)
        
        cv2.rectangle(mask, (shadow_x, shadow_y), 
                      (shadow_x + shadow_width, shadow_y + shadow_height), 
                      (0.5, 0.5, 0.5), -1) 
        
        augmented_img = (augmented_img.astype(np.float32) * mask).astype(np.uint8)

    if random.random() > 0.85:
        noise = np.random.normal(0, 25, augmented_img.shape).astype(np.uint8)
        augmented_img = cv2.add(augmented_img, noise)
    
    return augmented_img, augmented_angle

class DrivingDataset(Dataset):
    def __init__(self, samples, data_path, is_training=True):
        self.samples = samples
        self.data_path = data_path
        self.is_training = is_training
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        batch_sample = self.samples.iloc[idx]
        
        img_name = batch_sample['image_path'].strip()
        img_path = os.path.join(self.data_path, img_name)
        
        angular_velocity = float(batch_sample['angular_velocity_z'])
        linear_velocity = float(batch_sample['linear_velocity_x'])
        
        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        if self.is_training:
            img, angular_velocity = augment_image(img, angular_velocity)
        
        img = preprocess_image(img)
        
        img = (img / 255.0) - 0.5
        img = np.transpose(img, (2, 0, 1))
        
        img_tensor = torch.tensor(img, dtype=torch.float32)
        
        labels_tensor = torch.tensor([angular_velocity, linear_velocity], dtype=torch.float32)
        
        return img_tensor, labels_tensor
        

class ImprovedDave2Model(nn.Module):
    def __init__(self):
        super(ImprovedDave2Model, self).__init__()
        
        self.conv1 = nn.Conv2d(IMAGE_CHANNELS, 24, kernel_size=5, stride=2, padding=2) 
        self.bn1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(36)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.drop3 = nn.Dropout(0.2)
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.drop5 = nn.Dropout(0.3)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(5120, 100) 
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.drop_fc1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(100, 50)
        self.bn_fc2 = nn.BatchNorm1d(50)
        self.drop_fc2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(50, 10)
        self.drop_fc3 = nn.Dropout(0.2)
        
        self.output = nn.Linear(10, 2) 

    def forward(self, x_img):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x_img)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop5(F.relu(self.bn5(self.conv5(x))))
        
        x = self.flatten(x)
                
        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.drop_fc3(F.relu(self.fc3(x)))
        
        x = self.output(x)
        return x

def weighted_combined_loss(y_pred, y_true):
    mse_loss_fn = nn.MSELoss(reduction='none') 
    mae_loss_fn = nn.L1Loss(reduction='none') 
    
    mse_loss = mse_loss_fn(y_pred, y_true)
    mae_loss = mae_loss_fn(y_pred, y_true)

    angle_diff = torch.abs(y_true[:, 0] - y_pred[:, 0]).unsqueeze(1) 
    weight_multiplier = torch.where(angle_diff > 0.2, 2.0, 1.0)
    weights = torch.cat([10.0 * weight_multiplier, torch.ones_like(y_pred[:, 1]).unsqueeze(1)], dim=1) 
    weighted_loss = weights * (0.7 * mse_loss + 0.3 * mae_loss)
    
    return torch.mean(weighted_loss)


def balance_dataset(df):
    straight_data = df[abs(df['angular_velocity_z']) < 0.1]
    slight_turn_data = df[(abs(df['angular_velocity_z']) >= 0.1) & (abs(df['angular_velocity_z']) < 0.3)]
    sharp_turn_data = df[abs(df['angular_velocity_z']) >= 0.3]
    
    # print(f"데이터 분포:")
    # print(f"  직진: {len(straight_data)}개")
    # print(f"  약간 회전: {len(slight_turn_data)}개")
    # print(f"  급격한 회전: {len(sharp_turn_data)}개")
    
    sharp_turn_oversampled = pd.concat([sharp_turn_data] * 3, ignore_index=True) if len(sharp_turn_data) > 0 else sharp_turn_data
    slight_turn_oversampled = pd.concat([slight_turn_data] * 2, ignore_index=True) if len(slight_turn_data) > 0 else slight_turn_data
    
    balanced_df = pd.concat([straight_data, slight_turn_oversampled, sharp_turn_oversampled], ignore_index=True)
    
    # print(f"균형 조정 후 총 데이터: {len(balanced_df)}개")
    return balanced_df

def train_and_evaluate(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, start_epoch, EPOCHS, patience, FINAL_MODEL_SAVE_PATH):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n개선된 PyTorch 모델 훈련을 시작합니다 (Pure Image Regression)...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader): 
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images) 
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                mae = F.l1_loss(outputs[:, 0].unsqueeze(1), labels[:, 0].unsqueeze(1)) 
                
                val_running_loss += loss.item() * images.size(0)
                val_running_mae += mae.item() * images.size(0)
                
            val_loss = val_running_loss / len(validation_loader.dataset)
            val_mae = val_running_mae / len(validation_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val Omega_z MAE: {val_mae:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            print(f"  Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}). Saving model ...")
            torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
                    
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break
            
    print("\n훈련 종료. 최고 성능 모델 로드 및 저장.")
    model.load_state_dict(torch.load(FINAL_MODEL_SAVE_PATH))
    torch.save(model.state_dict(), 'dave2_model_final.pth')

    print(f"최고 성능 모델: {FINAL_MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f})")


if __name__ == '__main__':
    device = get_device()
    
    try:
        data_df = pd.read_csv(LOG_FILE_PATH)
    except FileNotFoundError:
        print(f"오류: '{LOG_FILE_PATH}' 파일을 찾을 수 없습니다.")
        exit()
        
    if 'linear_velocity_x' not in data_df.columns:
        print("경고: 'linear_velocity_x' 컬럼을 찾을 수 없습니다. 0.0으로 처리됩니다.")
        data_df['linear_velocity_x'] = 0.0
    if 'angular_velocity_z' not in data_df.columns and 'steering_angle' in data_df.columns:
        data_df.rename(columns={'steering_angle': 'angular_velocity_z'}, inplace=True)
        
    balanced_df = balance_dataset(data_df)
    train_samples, validation_samples = train_test_split(balanced_df, test_size=0.2, random_state=42)
    
    print(f"\n총 데이터 수: {len(data_df)}")
    print(f"훈련 데이터 수: {len(train_samples)}")
    print(f"검증 데이터 수: {len(validation_samples)}")
    
    train_dataset = DrivingDataset(train_samples, DATA_PATH, is_training=True)
    validation_dataset = DrivingDataset(validation_samples, DATA_PATH, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ImprovedDave2Model().to(device)
    print(model) 
    
    criterion = weighted_combined_loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    start_epoch = 0
    patience = 50
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=True
    )
    
    train_and_evaluate(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, start_epoch, EPOCHS, patience, FINAL_MODEL_SAVE_PATH)
