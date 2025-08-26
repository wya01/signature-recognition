# ✅ utils/inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import json

# === 图像转换器 ===
transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === 加载 label_map.json，并恢复 index → 原始文件夹编号 ===
with open("E:/MyProject/signature_project/experiments/resnet18_E7/label_map.json") as f:
    label_map = json.load(f)

label_list = [None] * len(label_map)
for folder_name, idx in label_map.items():
    label_list[idx] = folder_name  # 保证 index 对应的是真实编号字符串，如 "1", "2", ...

# === 分类模型加载（ResNet18） ===
def load_resnet18_classifier(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

classifier_model = load_resnet18_classifier(
    "E:/MyProject/signature_project/experiments/resnet18_E7/best_model.pth",
    num_classes=len(label_map)
)

# === 分类预测函数 ===
def predict_user(processed_img):
    rgb_img = Image.fromarray(processed_img).convert("RGB")
    x = transform_infer(rgb_img).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    with torch.no_grad():
        logits = classifier_model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

    user_id = label_list[pred_class]  # 返回原始文件夹编号（字符串形式）
    return user_id, confidence

# === Triplet 模型（用于真伪验证） ===
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

# === 加载 Triplet 模型 ===
siamese_model = DeeperCNN()
triplet_model_path = "E:/MyProject/signature_project/experiments/Triplet_E1(personsplit)/best_model.pth"
siamese_model.load_state_dict(torch.load(triplet_model_path, map_location="cpu"))
siamese_model.eval()

# === 真伪验证函数 ===
def verify_signature(img1, img2, threshold=2.480):
    to_tensor = transforms.ToTensor()
    x1 = to_tensor(img1).unsqueeze(0)  # [1, 1, H, W]
    x2 = to_tensor(img2).unsqueeze(0)
    with torch.no_grad():
        f1 = siamese_model(x1)
        f2 = siamese_model(x2)
        distance = F.pairwise_distance(f1, f2).item()
    return distance, threshold
