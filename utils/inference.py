import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# === 基础路径（项目根目录） ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # signature_project/
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === 图像转换器 ===
transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === label_map.json 加载 ===
LABEL_PATH = os.path.join(MODEL_DIR, "resnet18_E7", "label_map.json")
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)
else:
    print("⚠️ Warning: label_map.json not found, using dummy labels.")
    label_map = {"0": "User_01"}

label_list = [None] * len(label_map)
for folder_name, idx in label_map.items():
    label_list[idx] = folder_name  # 保证 index 对应的是真实编号字符串

# === 分类模型加载（ResNet18） ===
def load_resnet18_classifier():
    model_path = os.path.join(MODEL_DIR, "resnet18_E7", "best_model.pth")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# 延迟加载
_classifier_model = None
def get_classifier_model():
    global _classifier_model
    if _classifier_model is None:
        _classifier_model = load_resnet18_classifier()
    return _classifier_model

# === 分类预测函数 ===
def predict_user(processed_img):
    rgb_img = Image.fromarray(processed_img).convert("RGB")
    x = transform_infer(rgb_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    model = get_classifier_model()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

    user_id = label_list[pred_class]
    return user_id, confidence

# === Triplet 模型定义 ===
class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)

# === Triplet 模型加载 ===
def load_triplet_model():
    model_path = os.path.join(MODEL_DIR, "Triplet_E1(personsplit)", "best_model.pth")
    model = DeeperCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

_triplet_model = None
def get_triplet_model():
    global _triplet_model
    if _triplet_model is None:
        _triplet_model = load_triplet_model()
    return _triplet_model

# === 真伪验证函数 ===
def verify_signature(img1, img2, threshold=2.480):
    to_tensor = transforms.ToTensor()
    x1 = to_tensor(img1).unsqueeze(0)  # [1, 1, H, W]
    x2 = to_tensor(img2).unsqueeze(0)

    model = get_triplet_model()

    with torch.no_grad():
        f1 = model(x1)
        f2 = model(x2)
        distance = F.pairwise_distance(f1, f2).item()
    return distance, threshold
