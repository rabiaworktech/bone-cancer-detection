import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import os

# ─────────────────────────────────────────────
# CLIP SETUP
# ─────────────────────────────────────────────
try:
    import clip
    CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device="cpu")
    CLIP_MODEL.eval()

    XRAY_PROMPTS = clip.tokenize([
        "a medical bone X-ray scan",
        "a radiograph of human bones",
        "an X-ray image of a skeleton",
        "a bone X-ray showing skeletal structure",
        "a medical radiograph scan",
    ])

    NON_XRAY_PROMPTS = clip.tokenize([
        "a photograph of a person or animal",
        "a black and white photo",
        "a drawing, logo, or illustration",
        "a screenshot or document",
        "a nature or landscape photo",
        "a food or object photo",
        "a cartoon or graphic",
        "a portrait or selfie",
        "an artwork or painting",
        "a diagram or chart",
    ])

    CLIP_AVAILABLE = True
    print("CLIP loaded successfully.")
except Exception as e:
    CLIP_AVAILABLE = False
    print("CLIP not available:", e)


# ─────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────
CLASSES = ['Benign / Normal', 'Malignant / Tumor']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path="bone_cancer_model.pth"):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded.")
    else:
        print("Warning: using untrained model.")

    model = model.to(device)
    model.eval()
    return model


model = load_model()
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# PIXEL VALIDATION (RELAXED + NON-BLOCKING)
# ─────────────────────────────────────────────
def pixel_validation(image: Image.Image):
    img_rgb = np.array(image.convert('RGB')).astype(np.float32)
    img_gray = np.array(image.convert('L')).astype(np.float32)
    total = img_gray.size

    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    color_score = (np.mean(np.abs(r - g)) +
                   np.mean(np.abs(r - b)) +
                   np.mean(np.abs(g - b))) / 3.0
    if color_score > 35:
        return False, "Too much color"

    mean_brightness = np.mean(img_gray)
    if mean_brightness < 5 or mean_brightness > 210:
        return False, "Invalid brightness"

    # ⚠️ relaxed contrast (non-blocking)
    std_dev = np.std(img_gray)
    if std_dev < 15:
        return True, "Low contrast (allowed)"

    # histogram (relaxed)
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 255))
    dark_ratio = np.sum(hist[0:80]) / total
    bright_ratio = np.sum(hist[160:256]) / total
    mid_ratio = np.sum(hist[80:160]) / total

    if dark_ratio < 0.08:
        return False, "Too little dark region"
    if bright_ratio < 0.01:
        return False, "No bone highlights"
    if mid_ratio > 0.85:
        return True, "Mid-tone heavy (allowed)"

    # texture (relaxed + non-blocking)
    grad_x = np.abs(img_gray[:, 1:] - img_gray[:, :-1])
    grad_y = np.abs(img_gray[1:, :] - img_gray[:-1, :])
    mean_gradient = (np.mean(grad_x) + np.mean(grad_y)) / 2.0

    if mean_gradient < 2:
        return True, "Too smooth (allowed)"

    return True, ""


# ─────────────────────────────────────────────
# CLIP VALIDATION (BALANCED)
# ─────────────────────────────────────────────
def clip_validation(image: Image.Image):
    if not CLIP_AVAILABLE:
        return False, "CLIP unavailable"

    try:
        image_input = CLIP_PREPROCESS(image).unsqueeze(0)

        with torch.no_grad():
            image_features = CLIP_MODEL.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            xray_features = CLIP_MODEL.encode_text(XRAY_PROMPTS)
            xray_features = xray_features / xray_features.norm(dim=-1, keepdim=True)
            xray_sims = (image_features @ xray_features.T).squeeze(0)
            xray_score = float(xray_sims.softmax(dim=0).cpu().numpy().max())

            nonxray_features = CLIP_MODEL.encode_text(NON_XRAY_PROMPTS)
            nonxray_features = nonxray_features / nonxray_features.norm(dim=-1, keepdim=True)
            nonxray_sims = (image_features @ nonxray_features.T).squeeze(0)
            nonxray_score = float(nonxray_sims.softmax(dim=0).cpu().numpy().max())

        print(f"CLIP → X-ray: {xray_score:.3f} | Non-X-ray: {nonxray_score:.3f}")

        # ✅ balanced threshold
        if xray_score < 0.15:
            return False, "Low X-ray similarity"

        # ✅ reject if clearly non-xray
        if nonxray_score > xray_score * 1.05:
            return False, "Looks like non-X-ray"

        return True, ""

    except Exception as e:
        print("CLIP error:", e)
        return False, "CLIP failed"


# ─────────────────────────────────────────────
# FINAL VALIDATION (ROBUST)
# ─────────────────────────────────────────────
def is_valid_xray(image: Image.Image):
    pixel_pass, pixel_reason = pixel_validation(image)
    clip_pass, clip_reason = clip_validation(image)

    print("Pixel:", pixel_pass, pixel_reason)
    print("CLIP:", clip_pass, clip_reason)

    # ✅ Primary: CLIP decides
    if clip_pass:
        return True, ""

    # ✅ Secondary: fallback to pixel
    if pixel_pass:
        return True, ""

    return False, pixel_reason or clip_reason


# ─────────────────────────────────────────────
# INFERENCE + GRAD-CAM
# ─────────────────────────────────────────────
def get_prediction_and_cam(image: Image.Image):

    valid, reason = is_valid_xray(image)
    if not valid:
        raise ValueError(reason)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, pred_idx = torch.max(probs, 0)
        class_name = CLASSES[pred_idx.item()]

    targets = [ClassifierOutputTarget(pred_idx.item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    img_resized = image.resize((224, 224))
    img_float = np.float32(img_resized) / 255
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    cam_image = Image.fromarray(visualization)

    return class_name, confidence.item(), cam_image