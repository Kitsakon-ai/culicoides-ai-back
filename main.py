import base64
import io
import os
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


app = FastAPI(title="Sandfly AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


CLASS_NAMES: List[str] = [
    "guttifer",
    "mahasarakhamense",
    "oxystoma",
    "peregrinus",
]

SPECIES_TO_GENUS = {
    "guttifer": "Culicoides",
    "mahasarakhamense": "Culicoides",
    "oxystoma": "Culicoides",
    "peregrinus": "Culicoides",
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def build_taxonomy(species: str) -> Dict[str, str]:
    return {
        "kingdom": "Animalia",
        "phylum": "Arthropoda",
        "class": "Insecta",
        "order": "Diptera",
        "family": "Ceratopogonidae",
        "genus": SPECIES_TO_GENUS.get(species, "Unknown"),
        "species": species,
    }


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.50:
        return "low"
    return "ood"


def normalize_model_name(name: str) -> str:
    name = (name or "").strip().lower()

    aliases = {
        "efficientnet":            "efficientnet_b0",
        "efficientnetb0":          "efficientnet_b0",
        "efficientnet_b0":         "efficientnet_b0",
        "efficientnetb0_tif_best": "efficientnet_b0",
        "effb0":                   "efficientnet_b0",

        "resnet":        "resnet50",
        "resnet50":      "resnet50",
        "resnet50_best": "resnet50",
        "resnet50_tif_best": "resnet50",

        "densenet":              "densenet121",
        "densenet121":           "densenet121",
        "densenet121_best":      "densenet121",
        "densenet121_tif_best":  "densenet121",
    }

    return aliases.get(name, name)


def create_efficientnet_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def create_resnet_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def create_densenet_model(num_classes: int) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def load_model(path: str, model_type: str):
    full_path = os.path.join(BASE_DIR, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    if model_type == "efficientnet":
        loaded_model = create_efficientnet_model(len(CLASS_NAMES))
    elif model_type == "resnet":
        loaded_model = create_resnet_model(len(CLASS_NAMES))
    elif model_type == "densenet":
        loaded_model = create_densenet_model(len(CLASS_NAMES))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    state_dict = torch.load(full_path, map_location=DEVICE)
    loaded_model.load_state_dict(state_dict)
    loaded_model.to(DEVICE)
    loaded_model.eval()

    return loaded_model


MODELS: Dict[str, nn.Module] = {}


try:
    efficientnet_model = load_model("EfficientNetB0_tif_best.pth", "efficientnet")
    MODELS["efficientnet"] = efficientnet_model
    MODELS["efficientnet_b0"] = efficientnet_model
except Exception as e:
    print(f"[WARN] Failed to load EfficientNetB0_tif_best.pth: {e}")


try:
    resnet_model = load_model("ResNet50_tif_best.pth", "resnet")
    MODELS["resnet"] = resnet_model
    MODELS["resnet50"] = resnet_model
except Exception:
    try:
        resnet_model = load_model("ResNet50_best.pth", "resnet")
        MODELS["resnet"] = resnet_model
        MODELS["resnet50"] = resnet_model
    except Exception as e:
        raise RuntimeError(f"Failed to load ResNet50 model: {e}")


try:
    densenet_model = load_model("DenseNet121_tif_best.pth", "densenet")
    MODELS["densenet"] = densenet_model
    MODELS["densenet121"] = densenet_model
except Exception:
    try:
        densenet_model = load_model("DenseNet121_best.pth", "densenet")
        MODELS["densenet"] = densenet_model
        MODELS["densenet121"] = densenet_model
    except Exception as e:
        raise RuntimeError(f"Failed to load DenseNet121 model: {e}")


if not MODELS:
    raise RuntimeError("No model loaded successfully. Please check your model files.")


def read_image_from_upload(content: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


def pil_to_input_tensor(image: Image.Image) -> torch.Tensor:
    return transform(image).unsqueeze(0).to(DEVICE)


def predict_tensor(active_model: nn.Module, x: torch.Tensor):
    with torch.no_grad():
        logits = active_model(x)
        probs = F.softmax(logits, dim=1)[0]

    return logits, probs


def build_prediction_response(probs: torch.Tensor) -> Dict[str, Any]:
    probs_list = probs.detach().cpu().tolist()
    best_idx = int(torch.argmax(probs).item())

    species = CLASS_NAMES[best_idx]
    conf = float(probs[best_idx].item())

    top_k = [
        {
            "name": CLASS_NAMES[i],
            "probability": float(probs_list[i])
        }
        for i in range(len(CLASS_NAMES))
    ]

    top_k.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "species": species,
        "genus": SPECIES_TO_GENUS.get(species, "Unknown"),
        "confidence": conf,
        "topK": top_k,
        "confidenceLevel": confidence_level(conf),
        "taxonomy": build_taxonomy(species),
    }


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{encoded}"


def get_target_layer(active_model: nn.Module, model_name: str):
    model_name = normalize_model_name(model_name)

    if model_name == "efficientnet_b0":
        return active_model.features[-3]
    if model_name == "resnet50":
        return active_model.layer4[-1]
    if model_name == "densenet121":
        return active_model.features.denseblock4

    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def capture_activations_and_gradients(image: Image.Image, active_model: nn.Module, model_name: str):
    """Run one forward+backward pass and capture the target layer's activations
    and gradients, so Grad-CAM++ can be derived without re-running inference."""
    target_layer = get_target_layer(active_model, model_name)

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    x = pil_to_input_tensor(image)
    logits = active_model(x)
    class_idx = int(torch.argmax(logits, dim=1).item())

    active_model.zero_grad()
    logits[0, class_idx].backward()

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks failed.")

    act = activations[0][0]  # [C, H, W]
    grad = gradients[0][0]   # [C, H, W]
    return act, grad, class_idx


def gradcam_pp_weights(act: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Grad-CAM++ — weights each channel using second/third-order gradient terms
    so multiple/overlapping evidence regions are localized more precisely than
    plain Grad-CAM's global-average weighting. This replaces plain Grad-CAM."""
    eps = 1e-8

    grad2 = grad.pow(2)
    grad3 = grad2 * grad
    sum_act = act.sum(dim=(1, 2), keepdim=True)

    alpha_denom = grad2.mul(2) + sum_act * grad3
    alpha_denom = torch.where(
        alpha_denom != 0, alpha_denom, torch.full_like(alpha_denom, eps)
    )
    alphas = grad2 / alpha_denom

    weights = (alphas * torch.relu(grad)).sum(dim=(1, 2), keepdim=True)
    return (weights * act).sum(dim=0)


def normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam_np = cam.cpu().numpy()
    cam_np = cv2.resize(cam_np, (224, 224))
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    return cam_np


def cam_to_heatmap_base64(cam: torch.Tensor) -> str:
    """Pure heatmap — just the colorized CAM, with no original image blended in."""
    cam_np = normalize_cam(cam)

    fig, ax = plt.subplots()
    ax.imshow(cam_np, cmap="jet")
    ax.axis("off")

    return fig_to_base64(fig)


def cam_to_overlay_base64(cam: torch.Tensor, image: Image.Image) -> str:
    """Grad-CAM++ overlay — heatmap blended on top of the original image."""
    cam_np = normalize_cam(cam)

    original = image.resize((224, 224))
    original_np = np.array(original).astype(np.float32) / 255.0

    fig, ax = plt.subplots()
    ax.imshow(original_np)
    ax.imshow(cam_np, cmap="jet", alpha=0.6)
    ax.axis("off")

    return fig_to_base64(fig)


def make_gradcam(image: Image.Image, active_model: nn.Module, model_name: str):
    """Grad-CAM++ — the sole CAM method. Returns (heatmap_base64, overlay_base64,
    class_idx) from a single forward+backward pass: the pure heatmap and the
    heatmap overlaid on the original image."""
    act, grad, class_idx = capture_activations_and_gradients(image, active_model, model_name)
    cam = gradcam_pp_weights(act, grad)

    heatmap_img = cam_to_heatmap_base64(cam)
    overlay_img = cam_to_overlay_base64(cam, image)

    return heatmap_img, overlay_img, class_idx


@app.get("/")
def root():
    return {
        "message": "Sandfly API running",
        "available_models": list(MODELS.keys()),
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "normalized_models": ["efficientnet", "resnet", "densenet"],
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": list(MODELS.keys()),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    ml_model: str = Form("efficientnet"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    ml_model = normalize_model_name(ml_model)

    active_model = MODELS.get(ml_model)

    if active_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ml_model: {ml_model}. Available: {list(MODELS.keys())}"
        )

    content = await file.read()
    image = read_image_from_upload(content)

    try:
        x = pil_to_input_tensor(image)
        _, probs = predict_tensor(active_model, x)
        result = build_prediction_response(probs)

        return {
            **result,
            "modelUsed": ml_model,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        ) from exc


@app.post("/predict-with-gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...),
    ml_model: str = Form("efficientnet"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    ml_model = normalize_model_name(ml_model)

    active_model = MODELS.get(ml_model)

    if active_model is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ml_model: {ml_model}. Available: {list(MODELS.keys())}"
        )

    content = await file.read()
    image = read_image_from_upload(content)

    try:
        x = pil_to_input_tensor(image)
        _, probs = predict_tensor(active_model, x)
        result = build_prediction_response(probs)

        heatmap_image, gradcam_image, _ = make_gradcam(image, active_model, ml_model)

        return {
            **result,
            "gradcam": gradcam_image,
            "heatmap": heatmap_image,
            "modelUsed": ml_model,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference with Grad-CAM failed: {str(exc)}"
        ) from exc