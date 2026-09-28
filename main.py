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
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
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
    if confidence >= 0.45:
        return "low"
    # ต่ำกว่า 0.45 = โมเดลไม่มั่นใจว่าเป็นปีก Culicoides → ไม่แสดงผลจำแนก
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
        print(f"[WARN] Failed to load ResNet50 model: {e}")


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
        print(f"[WARN] Failed to load DenseNet121 model: {e}")


# โมเดลภาพเต็มชุดเดิมเป็นของเสริมแล้ว ไม่ใช่ของบังคับ
# ระบบเลิกใช้มันเป็นตัวตัดสินแล้ว — ตัวตัดสินคือโมเดลครอปใน /detect
# ถ้าไม่มีไฟล์ก็ปล่อยให้แอปขึ้นได้ตามปกติ แค่ /predict จะตอบ 503
if not MODELS:
    # ASCII only: stdout บน Windows/container บางตัวเป็น cp1252 ถ้าพิมพ์ไทยจะ crash ตอนบูต
    print("[WARN] no whole-image model loaded - /predict and /predict-with-gradcam will return 503")


# ======================================================================
# ด่าน kNN — ปฏิเสธภาพที่ไม่ใช่ปีก Culicoides ก่อนตอบ
# ----------------------------------------------------------------------
# max-softmax บอกได้แค่ "ในบรรดา 4 คลาสนี้ อันไหนน่าจะใช่ที่สุด" ไม่ได้บอกว่า
# ภาพนี้เป็นปีกจริงหรือเปล่า วัดจริงแล้วภาพพื้นหลังเปล่า ๆ ที่ไม่มีอะไรอยู่เลย
# ยังได้ความมั่นใจ 66-79% และถูกตอบเป็นสปีชีส์
#
# ด่านนี้วัดระยะ cosine จาก feature ของภาพขาเข้า ไปหาภาพเทรนที่ใกล้ที่สุด
# ไกลเกินเกณฑ์ = ไม่เคยเห็นอะไรแบบนี้ = ตอบว่าไม่รู้
# วัดบน test 40 ภาพจริง + 200 ภาพที่ไม่ควรตอบ: ภาพจริงผ่าน 100% ของแปลกหลุด 0%
#
# คลังอ้างอิงกับเกณฑ์มาจาก calibrate_backend_gate.py (เลือกจาก val เท่านั้น)
# *** ผูกกับน้ำหนักโมเดลชุดนี้โดยตรง ถ้าเปลี่ยนไฟล์ .pth ต้องสร้างคลังใหม่ ***
# ======================================================================

KNN_BANK_FILES = {
    "efficientnet_b0": "knn_bank_effb0.npz",
    "resnet50": "knn_bank_resnet50.npz",
    "densenet121": "knn_bank_densenet121.npz",
}

KNN_GATES: Dict[str, Dict[str, Any]] = {}
_FEATURE_BUFFER: Dict[str, torch.Tensor] = {}


def _head_linear(active_model: nn.Module, key: str) -> nn.Module:
    """ชั้น Linear สุดท้าย — อินพุตของมันคือ feature ที่ใช้วัดระยะ"""
    if key == "resnet50":
        return active_model.fc[1]
    return active_model.classifier[1]


def attach_knn_gate(key: str, active_model: nn.Module) -> None:
    path = os.path.join(BASE_DIR, KNN_BANK_FILES[key])

    if not os.path.exists(path):
        print(f"[WARN] kNN bank not found: {path} - gate disabled for {key}")
        return

    data = np.load(path, allow_pickle=True)

    KNN_GATES[key] = {
        "bank": torch.from_numpy(data["bank"]).float().to(DEVICE),
        "k": int(data["k"]),
        "threshold": float(data["threshold"]),
    }

    _head_linear(active_model, key).register_forward_hook(
        lambda module, inp, out, key=key: _FEATURE_BUFFER.__setitem__(key, inp[0].detach())
    )

    print(f"[OK] kNN gate ready for {key} "
          f"(k={KNN_GATES[key]['k']}, threshold={KNN_GATES[key]['threshold']:.4f})")


for _key in ("efficientnet_b0", "resnet50", "densenet121"):
    if _key in MODELS:
        attach_knn_gate(_key, MODELS[_key])


def knn_gate_result(key: str):
    """คืน (คะแนน, ผ่านไหม) จาก feature ของ forward ครั้งล่าสุดของโมเดลนั้น

    ไม่มีคลังอ้างอิง -> คืน (None, True) คือไม่ปิดกั้น ระบบทำงานเหมือนเดิมทุกอย่าง
    """
    cfg = KNN_GATES.get(key)
    feat = _FEATURE_BUFFER.get(key)

    if cfg is None or feat is None:
        return None, True

    with torch.no_grad():
        f = feat / torch.linalg.vector_norm(feat, dim=1, keepdim=True).clamp_min(1e-9)
        dist = 1.0 - f @ cfg["bank"].t()
        near = torch.topk(dist, cfg["k"], dim=1, largest=False).values
        score = float((-near.mean(dim=1))[0])

    return score, score >= cfg["threshold"]


def apply_knn_gate(result: Dict[str, Any], key: str) -> Dict[str, Any]:
    """เติมผลของด่านลงใน response

    ถ้าไม่ผ่าน บังคับ confidenceLevel เป็น "ood" ซึ่งหน้าเว็บรองรับอยู่แล้ว
    จึงไม่ต้องแก้ฝั่งหน้าบ้านเลย
    """
    score, accepted = knn_gate_result(key)

    if score is None:
        return result

    result["knnScore"] = round(score, 4)
    result["accepted"] = bool(accepted)

    if not accepted:
        result["confidenceLevel"] = "ood"
        result["oodReason"] = "ภาพนี้ไม่เหมือนปีกที่โมเดลเคยเรียนรู้ ระบบจึงไม่ระบุชนิด"

    return result



# ======================================================================
# โมเดลครอป — ใช้กับ /detect เท่านั้น ("1 รูปมีหลายปีก ตอบว่ามีกี่ชนิด")
# ----------------------------------------------------------------------
# แยกขาดจากโมเดลเดิมของ /predict ทุกอย่าง: คนละน้ำหนัก คนละคลัง kNN
# คนละไปป์ไลน์ภาพ (ต้องครอปปีกก่อนเข้าโมเดล) จึงไม่กระทบ endpoint เดิมเลย
#
# import แบบกันพัง — ถ้าไม่มี scipy หรือไฟล์โมเดลครอปไม่ครบ ให้ /detect
# ตอบ 503 ไปเลย แต่ /predict กับ /predict-with-gradcam ต้องทำงานได้ตามปกติ
# ======================================================================

# ── ทำไมค่าเริ่มต้นเป็น EfficientNet-B0 ────────────────────────────────
# วัดจริงบน test set 40 ภาพ (results_masked/crop_banks.csv จาก build_crop_banks.py):
#
#   โมเดล             ความแม่น  ภาพจริงผ่านด่าน  ของปลอมหลุด  ขนาดไฟล์
#   EfficientNet-B0     100%        97.5%          1.25%       15.6 MB
#   DenseNet121         100%       100.0%          2.50%       27.1 MB
#   ResNet50            100%        97.5%          1.88%       90.0 MB
#
# ความแม่นเท่ากันหมด (100% ทั้งสามตัว) จึงตัดสินกันที่ด่าน kNN แทน:
#   1. ของปลอมหลุดน้อยที่สุด (1.25%) — สำคัญที่สุดสำหรับระบบนี้ เพราะต้องกล้า
#      ตอบ "ไม่รู้" เมื่อภาพไม่ใช่ปีก Culicoides ดีกว่าเดาแล้วผู้ใช้เชื่อผิด
#   2. ไฟล์เล็กที่สุด — ResNet50 ใหญ่กว่า 5.8 เท่าโดยไม่ได้อะไรคืนมา
#      (แม่นเท่ากัน ของปลอมหลุดมากกว่า)
#
# ข้อแลกเปลี่ยนที่ยอมรับ: ปฏิเสธปีกจริงไป 2.5% ขณะที่ DenseNet121 รับครบ 100%
# ถ้าให้ความสำคัญกับ "ห้ามปฏิเสธปีกจริง" มากกว่า ให้เปลี่ยนเป็น "densenet121"
#
# ข้อจำกัด: test set มีแค่ 40 ภาพ และทั้งสามตัวได้ 100% จึงยังแยกไม่ออกว่า
# ใครเก่งกว่าใครจริง ๆ ตัวเลขนี้บอกได้แค่ว่า "ไม่มีตัวไหนแย่"
DETECT_ARCH = "effb0"               # ตัวเริ่มต้นเมื่อไม่ระบุ ml_model
DETECT_GATES: Dict[str, Any] = {}   # arch -> gate ที่โหลดสำเร็จ
DETECT_GATE: Dict[str, Any] = None  # gate ของ DETECT_ARCH (ตัวเริ่มต้น)
DETECT_ERROR: str = None
wing_detect = None

# ชื่อที่หน้าเว็บส่งมา -> ชื่อสถาปัตยกรรมของโมเดลครอป
DETECT_ALIASES = {
    "efficientnet": "effb0", "efficientnetb0": "effb0",
    "efficientnet_b0": "effb0", "effb0": "effb0",
    "densenet": "densenet121", "densenet121": "densenet121",
    "resnet": "resnet50", "resnet50": "resnet50",
    "ensemble": "ensemble", "all": "ensemble",
}

try:
    import wing_detect as _wd

    os.environ.setdefault("CROP_MODEL_DIR", BASE_DIR)
    archs = _wd.available_archs()
    if DETECT_ARCH not in archs:
        raise FileNotFoundError(
            f"crop model files for {DETECT_ARCH} not found in {_wd.model_dir()}")

    wing_detect = _wd
    # โหลดทุกตัวที่ไฟล์ครบ — มีหลายตัวถึงจะเทียบกันได้
    # แต่ละตัวมีรัศมี kNN ของตัวเอง ใช้ข้ามกันไม่ได้ (ต่างกันเกือบ 4 เท่า)
    for _a in archs:
        DETECT_GATES[_a] = _wd.load_gate(_a)
        g = DETECT_GATES[_a]
        print(f"[OK] detect gate ready: {g['name']} "
              f"(k={g['k']}, radius={g['radius']:.4f}, bank={g['bank'].shape[0]})")
    DETECT_GATE = DETECT_GATES[DETECT_ARCH]
except Exception as exc:  # noqa: BLE001 - ตั้งใจกลืนทุก error ไม่ให้แอปล้ม
    DETECT_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[WARN] /detect disabled - {DETECT_ERROR}")


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
        return active_model.features[-1]
    if model_name == "resnet50":
        return active_model.layer4[-1]
    if model_name == "densenet121":
        return active_model.features.norm5

    raise ValueError(f"Unsupported model for Grad-CAM: {model_name}")


def capture_activations_and_gradients(image: Image.Image, active_model: nn.Module, model_name: str):
    """Run one forward+backward pass and capture the target layer's activations
    and gradients, so Grad-CAM++ can be derived without re-running inference."""
    target_layer = get_target_layer(active_model, model_name)

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())
        out.register_hook(lambda grad: gradients.append(grad.detach()))

    fh = target_layer.register_forward_hook(forward_hook)

    x = pil_to_input_tensor(image)
    logits = active_model(x)
    class_idx = int(torch.argmax(logits, dim=1).item())

    active_model.zero_grad()
    logits[0, class_idx].backward()

    fh.remove()

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

    original = transforms.functional.center_crop(
        transforms.functional.resize(image, (256, 256)), 224
    )
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
        "knnGate": {k: {"k": v["k"], "threshold": round(v["threshold"], 4),
                        "bankSize": int(v["bank"].shape[0])}
                    for k, v in KNN_GATES.items()},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": list(MODELS.keys()),
        "classes": CLASS_NAMES,
        "knn_gate_enabled": sorted(KNN_GATES.keys()),
        "detect": {
            "ready": DETECT_GATE is not None,
            "arch": DETECT_ARCH if DETECT_GATE is not None else None,
            "error": DETECT_ERROR,
            "k": DETECT_GATE["k"] if DETECT_GATE else None,
            "radius": round(DETECT_GATE["radius"], 4) if DETECT_GATE else None,
            "bankSize": int(DETECT_GATE["bank"].shape[0]) if DETECT_GATE else None,
            "frame": list(DETECT_GATE["frame"]) if DETECT_GATE else None,
            "padFrac": DETECT_GATE["pad_frac"] if DETECT_GATE else None,
            # แต่ละสถาปัตยกรรมมีรัศมี kNN ของตัวเอง ใช้ข้ามกันไม่ได้
            "models": [
                {"arch": a, "name": g["name"], "k": g["k"],
                 "radius": round(g["radius"], 4), "bankSize": int(g["bank"].shape[0])}
                for a, g in DETECT_GATES.items()
            ],
        },
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    ml_model: str = Form("efficientnet"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    if not MODELS:
        raise HTTPException(
            status_code=503,
            detail="Whole-image models are no longer bundled. Use /detect instead.")

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
        result = apply_knn_gate(result, ml_model)

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

    if not MODELS:
        raise HTTPException(
            status_code=503,
            detail="Whole-image models are no longer bundled. Use /detect instead.")

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
        result = apply_knn_gate(result, ml_model)

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


# ======================================================================
# /detect — ตรวจปีกทุกตัวในภาพเดียว แล้วตอบว่ามีกี่ชนิด
# ----------------------------------------------------------------------
# ต่างจาก /predict ที่มองทั้งภาพเป็นปีกเดียว: ตัวนี้หาปีกทุกตัวก่อน
# แล้วครอปทีละตัวเข้าโมเดลครอป พร้อมด่าน kNN ของตัวเอง
# พิกัด box ที่คืนเป็นพิกัดบน "ภาพต้นฉบับ" หน้าบ้านเอาไปวาดกรอบได้เลย
# ======================================================================

DETECT_MODEL_LABEL = {"effb0": "efficientnet",
                      "densenet121": "densenet",
                      "resnet50": "resnet"}


def pil_to_data_url(image: Image.Image, quality: int = 88) -> str:
    """ภาพปีกที่ครอปแล้ว → data URL

    ใช้ JPEG ไม่ใช่ PNG เพราะต้องส่งหลายใบต่อหนึ่ง request
    (ปีก 4 ตัว = crop 4 + heatmap 4 + overlay 4) PNG จะทำให้ response บวมเกิน
    ลิมิต 4.5 MB ของ Vercel ได้ง่าย ๆ
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    gradcam: str = Form("false"),
    ml_model: str = Form("efficientnet"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Grad-CAM ต่อปีกเป็นของเสริม ปิดไว้เป็นค่าเริ่มต้น
    # เปิดแล้วจะช้าขึ้นและ response ใหญ่ขึ้นมาก (ภาพ base64 ต่อปีกหนึ่งใบ)
    want_cam = str(gradcam).strip().lower() in {"1", "true", "yes", "on"}

    if DETECT_GATE is None:
        raise HTTPException(
            status_code=503,
            detail=f"Detect endpoint unavailable: {DETECT_ERROR}")

    raw_key = str(ml_model).strip().lower()
    key = DETECT_ALIASES.get(raw_key, raw_key)
    if key == "ensemble":
        gates = list(DETECT_GATES.values())
    else:
        one = DETECT_GATES.get(key)
        if one is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown ml_model for /detect: {ml_model}. "
                       f"Available: {sorted(DETECT_GATES)} + 'ensemble'")
        gates = [one]

    # Grad-CAM ทำด้วยโมเดลเดียวพอ ไม่งั้น ensemble จะได้ภาพ 3 เท่า
    cam_gate = DETECT_GATE if DETECT_GATE in gates else gates[0]

    content = await file.read()
    image = read_image_from_upload(content)
    arr = np.asarray(image)
    height, width = arr.shape[:2]

    try:
        boxes, _bg = wing_detect.find_wings(arr)

        short = gates[0]["short"]
        wings = []
        for box in boxes:
            crop = wing_detect.crop_for_model(arr, box)

            # ให้ทุกโมเดลที่เลือกไว้ตัดสินปีกตัวนี้ แล้วค่อยรวมเป็นคำตอบเดียว
            results = [wing_detect.predict_wing(g, crop) for g in gates]
            species, why = wing_detect.combine(results, len(gates))

            # ความมั่นใจของคำตอบรวม = เฉลี่ยเฉพาะโมเดลที่ผ่านด่าน "และ" โหวตชนิดนี้
            voters = [r for r in results
                      if species and r["accepted"] and r["species"] == species]
            confidence = float(np.mean([r["confidence"] for r in voters])) if voters else None

            # Top-K เฉลี่ยความน่าจะเป็นข้ามทุกโมเดลที่ใช้ — ถ้าใช้ตัวเดียวก็คือของตัวนั้น
            mean_probs = np.mean([r["probs"] for r in results], axis=0)
            top_k = sorted(
                ({"name": short[i], "probability": float(pr)}
                 for i, pr in enumerate(mean_probs)),
                key=lambda item: item["probability"],
                reverse=True,
            )

            entry = {
                "box": [int(v) for v in box],
                "species": species,
                # ปีกที่ถูกปฏิเสธไม่มีชนิด จึงไม่มีความมั่นใจที่มีความหมาย
                "confidence": round(confidence, 4) if confidence is not None else None,
                "accepted": species is not None,
                # ค่าด่านของโมเดลตัวแรกที่เลือกใช้ ไว้ดูเร็ว ๆ — รายตัวครบอยู่ใน models
                # เพราะแต่ละสถาปัตยกรรมมีรัศมีของตัวเอง ใช้ข้ามกันไม่ได้
                "knnDistance": round(results[0]["distance"], 4),
                "knnRadius": round(results[0]["radius"], 4),
                "topK": top_k,
                "genus": SPECIES_TO_GENUS.get(species, "Unknown") if species else None,
                "taxonomy": build_taxonomy(species) if species else None,
                "consensus": why,
                "models": [
                    {
                        "arch": g["arch"],
                        "name": g["name"],
                        "species": r["species"],
                        "confidence": round(r["confidence"], 4) if r["accepted"] else None,
                        "accepted": r["accepted"],
                        "knnDistance": round(r["distance"], 4),
                        "knnRadius": round(r["radius"], 4),
                        "isWinner": bool(species and r["accepted"] and r["species"] == species),
                        # กราฟเปรียบเทียบโมเดลบนหน้าเว็บวาดจาก topK รายตัว
                        "topK": sorted(
                            ({"name": short[i], "probability": float(pr)}
                             for i, pr in enumerate(r["probs"])),
                            key=lambda item: item["probability"],
                            reverse=True,
                        ),
                    }
                    for g, r in zip(gates, results)
                ],
                # ภาพปีกที่ครอปแล้ว — เป็นภาพที่โมเดลเห็นจริง หน้าเว็บส่งต่อให้ LLM
                # อธิบายเฉพาะปีกตัวนี้ได้ แทนที่จะส่งทั้งสไลด์ไปให้มันงง
                "crop": pil_to_data_url(crop),
            }

            # โมเดลครอปทั้ง 3 ตัวเป็นสถาปัตยกรรมที่ get_target_layer รองรับอยู่แล้ว
            # และ transform ของ wing_detect ตรงกับของไฟล์นี้ทุกขั้น จึงใช้ซ้ำได้เลย
            # ทำ Grad-CAM ด้วยโมเดลเดียวพอ ไม่งั้น ensemble จะได้ภาพ 3 เท่า
            if want_cam and species:
                try:
                    heat, overlay, _ = make_gradcam(crop, cam_gate["model"], cam_gate["arch"])
                    entry["gradcam"] = overlay
                    entry["heatmap"] = heat
                    entry["gradcamModel"] = cam_gate["name"]
                except Exception as cam_exc:  # noqa: BLE001
                    # Grad-CAM พังไม่ควรทำให้ผลตรวจทั้งภาพพังไปด้วย
                    entry["gradcamError"] = str(cam_exc)

            wings.append(entry)

        named = [w["species"] for w in wings if w["species"]]
        species_found = sorted(set(named))

        # หาปีกไม่เจอเลย = ตอบ 0 ชนิด ไม่ใช่ error (เช่นอัปโหลด screenshot มา)
        return {
            "wings": wings,
            "speciesFound": species_found,
            "speciesCount": len(species_found),
            "wingsDetected": len(wings),
            "wingsAnswered": len(named),
            "imageSize": [int(width), int(height)],
            "modelUsed": "ensemble" if key == "ensemble"
                         else DETECT_MODEL_LABEL.get(key, key),
            "modelsUsed": [g["name"] for g in gates],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(exc)}"
        ) from exc

# ======================================================================
# รันตรง ๆ ด้วย `python main.py` ได้โดยไม่ต้องมี Dockerfile
# ----------------------------------------------------------------------
# ใช้กับโฮสต์ที่สั่ง `python app.py` เอง (เช่น HF Space SDK gradio/streamlit
# ซึ่งใช้ได้ฟรีบน CPU Basic) และใช้ทดสอบในเครื่องเร็ว ๆ
# อ่านพอร์ตจาก $PORT เพราะ Render/Railway/Cloud Run กำหนดพอร์ตมาให้เอง
# ======================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
