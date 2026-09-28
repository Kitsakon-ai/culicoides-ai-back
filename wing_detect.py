r"""
wing_detect.py — โมดูลหาปีกและตัดสินชนิด สำหรับให้ backend import ไปใช้
=========================================================================
ย้ายโค้ดมาจาก check_image.py ตรง ๆ ไม่เปลี่ยน logic ใด ๆ
ต่างจาก check_image.py แค่ 3 อย่าง:
  1. เป็นโมดูล import ได้ ไม่พิมพ์อะไรตอน import และไม่อ่าน sys.argv
  2. ค่าคงที่ pad_frac / frame / k / threshold อ่านจากไฟล์ .npz ไม่ hardcode
  3. รายชื่อคลาสอ่านจาก .npz ด้วย จึงไม่ต้องมี classes.txt ตอน deploy

ใช้แค่ numpy, scipy.ndimage, PIL, torch, torchvision — ไม่ใช้ cv2

ฟังก์ชันหลัก
  find_wings(arr)            -> (รายการกรอบ (x0,y0,x1,y1) เรียงซ้ายไปขวา, สีพื้นหลัง RGB)
  crop_for_model(arr, box)   -> PIL.Image ขนาดตาม frame ในคลัง (491x368)
  load_gate(arch)            -> dict: model, feat_fn, head_fn, bank, k, threshold, ...
  predict_wing(gate, img)    -> dict ผลของปีกหนึ่งตัวจากโมเดลหนึ่งตัว (ตัวช่วยสำหรับ backend)

ทดสอบรูปเดียว:
  mlpy wing_detect.py "path\to\image.jpg"
  mlpy wing_detect.py "split_dataset_nocrop\test\guttifer\xxx.tif"
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from scipy import ndimage
from torchvision import models, transforms

Image.MAX_IMAGE_PIXELS = None
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# สัดส่วนพื้นที่ขั้นต่ำของก้อนที่จะนับว่าเป็นปีก — เป็นพารามิเตอร์ของตัวหาปีก
# ไม่ได้ผูกกับโมเดล จึงไม่ได้เก็บไว้ในคลัง kNN
MIN_AREA = 0.002

# ชื่อไฟล์ของแต่ละสถาปัตยกรรม (ชุดเดียวกับ check_image.py)
ARCH_FILES = {
    "effb0":       ("EfficientNetB0_crop_best.pth", "knn_bank_effb0_crop.npz"),
    "densenet121": ("DenseNet121_crop_best.pth",    "knn_bank_densenet121_crop.npz"),
    "resnet50":    ("ResNet50_crop_best.pth",       "knn_bank_resnet50_crop.npz"),
}
ARCH_NAMES = {"effb0": "EfficientNet-B0", "densenet121": "DenseNet121",
              "resnet50": "ResNet50"}

TF = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEOM_CACHE = None


def model_dir():
    """โฟลเดอร์ที่เก็บ .pth กับ .npz

    รองรับทั้งสองรูปแบบการวางไฟล์:
      - ตอนวิจัย  trainmodel/crop_model/*.pth
      - ตอน deploy  culicoides-ai-back/*.pth  (วางไว้ข้างโมดูลเลย)
    ตั้ง CROP_MODEL_DIR ทับได้
    """
    env = os.environ.get("CROP_MODEL_DIR")
    if env:
        return env
    sub = os.path.join(_HERE, "crop_model")
    return sub if os.path.isdir(sub) else _HERE


def bank_path(arch):
    return os.path.join(model_dir(), ARCH_FILES[arch][1])


def ckpt_path(arch):
    return os.path.join(model_dir(), ARCH_FILES[arch][0])


def available_archs():
    """สถาปัตยกรรมที่มีไฟล์ครบทั้ง .pth และ .npz"""
    return [a for a in ARCH_FILES
            if os.path.exists(ckpt_path(a)) and os.path.exists(bank_path(a))]


def geometry():
    """(pad_frac, frame_w, frame_h) อ่านจากคลังใบแรกที่เจอ

    ทุกสถาปัตยกรรมถูกสร้างด้วยไปป์ไลน์ครอปชุดเดียวกัน ค่าจึงตรงกันเสมอ
    """
    global _GEOM_CACHE
    if _GEOM_CACHE is None:
        archs = available_archs()
        if not archs:
            raise FileNotFoundError(f"ไม่พบคลัง kNN ใน {model_dir()}")
        z = np.load(bank_path(archs[0]), allow_pickle=True)
        fw, fh = (int(v) for v in z["frame"])
        _GEOM_CACHE = (float(z["pad_frac"]), fw, fh)
    return _GEOM_CACHE


# ====================================================== ตัวหาปีก
def find_wings(arr):
    """คืน (รายการกรอบปีกทุกตัว เรียงซ้ายไปขวา, สีพื้นหลัง RGB)

    โค้ดเดียวกับ check_image.find_wings ทุกบรรทัด ต่างแค่คืนสีพื้นหลังเป็น RGB
    (check_image คืนค่าเฉลี่ยไปเลย เพราะใช้แค่พิมพ์)

    arr : numpy uint8 (H, W, 3)
    """
    gray = arr.mean(axis=2)
    h, w = gray.shape
    sl = [(slice(0, h // 8), slice(0, w // 8)), (slice(0, h // 8), slice(-w // 8, None)),
          (slice(-h // 8, None), slice(0, w // 8)), (slice(-h // 8, None), slice(-w // 8, None))]
    cp = np.concatenate([arr[a, b].reshape(-1, 3) for a, b in sl])
    bg = np.median(cp, axis=0)
    m = ndimage.binary_fill_holes(
        ndimage.binary_closing(gray < bg.mean() - 9, np.ones((7, 7))))
    lab, n = ndimage.label(m)
    if n == 0:
        return [], bg
    sizes = np.bincount(lab.ravel())
    boxes = []
    for i in range(1, n + 1):
        if sizes[i] < MIN_AREA * m.size:
            continue
        yy, xx = np.nonzero(lab == i)
        cw, ch = xx.max() - xx.min() + 1, yy.max() - yy.min() + 1
        # กันก้อนที่กินเกือบทั้งภาพ (สแกนทั้งสไลด์) และก้อนที่ผอม/ยาวผิดสัดส่วนปีก
        if cw > 0.92 * w or ch > 0.92 * h or not (0.25 <= cw / ch <= 4.0):
            continue
        boxes.append((int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())))
    boxes.sort(key=lambda b: b[0])
    return boxes, bg


def crop_for_model(arr, box, pad_frac=None, frame=None):
    """ครอปปีกหนึ่งตัวออกมาเป็นภาพขนาดที่โมเดลรับ

    pad_frac / frame ถ้าไม่ส่งมา จะอ่านจากคลัง kNN — ห้าม hardcode
    เพราะคลังถูกสร้างด้วยค่าชุดนี้ ถ้าครอปคนละแบบ ระยะ kNN จะเพี้ยนทันที
    """
    if pad_frac is None or frame is None:
        pf, fw, fh = geometry()
        pad_frac = pf if pad_frac is None else pad_frac
        frame = (fw, fh) if frame is None else frame
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = box
    p = int(max(x1 - x0, y1 - y0) * pad_frac)
    x0, y0 = max(0, x0 - p), max(0, y0 - p)
    x1, y1 = min(W, x1 + 1 + p), min(H, y1 + 1 + p)
    return Image.fromarray(arr[y0:y1, x0:x1]).resize(tuple(frame), Image.BICUBIC)


# ====================================================== โมเดล + ด่าน kNN
def _build(arch, ckpt, n_classes):
    """สร้างโมเดลและฟังก์ชันสกัด feature — ต้องตรงกับ build_crop_banks.py เป๊ะ

    DenseNet ต้องผ่าน relu ก่อน avgpool ตาม forward ของ torchvision
    ถ้าข้ามไป feature จะผิดและระยะ kNN ใช้ไม่ได้
    """
    if arch == "effb0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, n_classes))
        m.load_state_dict(torch.load(ckpt, map_location=DEV))
        feat = lambda x: m.avgpool(m.features(x)).flatten(1)
        head = lambda f: m.classifier(f)
    elif arch == "densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, n_classes))
        m.load_state_dict(torch.load(ckpt, map_location=DEV))
        feat = lambda x: torch.flatten(
            Fn.adaptive_avg_pool2d(Fn.relu(m.features(x), inplace=True), (1, 1)), 1)
        head = lambda f: m.classifier(f)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, n_classes))
        m.load_state_dict(torch.load(ckpt, map_location=DEV))

        def feat(x):
            y = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return torch.flatten(m.avgpool(m.layer4(m.layer3(m.layer2(m.layer1(y))))), 1)
        head = lambda f: m.fc(f)
    else:
        raise ValueError(f"ไม่รู้จักสถาปัตยกรรม {arch}")
    return m.to(DEV).eval(), feat, head


def load_gate(arch):
    """โหลดโมเดล + คลัง kNN ของสถาปัตยกรรมหนึ่งตัว

    คืน dict:
      model, feat_fn, head_fn      ตัวโมเดลและฟังก์ชันสกัด/หัวจำแนก
      bank, labels                 คลัง feature ที่ normalize แล้ว และป้ายคลาส
      k, threshold, radius         ค่าด่าน (radius = -threshold ใช้เทียบกับระยะตรง ๆ)
      classes, short               ชื่อคลาสเต็มและชื่อย่อ
      arch, name, pad_frac, frame  ข้อมูลประกอบ
    """
    cp, bp = ckpt_path(arch), bank_path(arch)
    if not (os.path.exists(cp) and os.path.exists(bp)):
        raise FileNotFoundError(f"ไฟล์ไม่ครบสำหรับ {arch}: {cp} / {bp}")
    z = np.load(bp, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    model, feat_fn, head_fn = _build(arch, cp, len(classes))
    fw, fh = (int(v) for v in z["frame"])
    return {
        "arch": arch,
        "name": ARCH_NAMES.get(arch, arch),
        "model": model,
        "feat_fn": feat_fn,
        "head_fn": head_fn,
        "bank": z["bank"],
        "labels": z["labels"],
        "k": int(z["k"]),
        "threshold": float(z["threshold"]),
        "radius": -float(z["threshold"]),
        "classes": classes,
        "short": [c.replace(" 4X", "") for c in classes],
        "pad_frac": float(z["pad_frac"]),
        "frame": (fw, fh),
    }


@torch.no_grad()
def predict_wing(gate, img):
    """ตัดสินปีกหนึ่งตัวด้วยโมเดลหนึ่งตัว

    img : PIL.Image ที่ผ่าน crop_for_model มาแล้ว
    คืน dict: accepted, species, confidence, distance, radius, probs

    ลำดับเดียวกับ check_image.py: สกัด feature -> softmax จากหัวจำแนก ->
    ระยะโคไซน์ถึงเพื่อนบ้านลำดับที่ k ในคลัง -> ผ่านเมื่อระยะ <= รัศมี
    """
    x = TF(img).unsqueeze(0).to(DEV)
    f = gate["feat_fn"](x)
    prob = torch.softmax(gate["head_fn"](f), 1)[0].cpu().numpy()
    fn = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
    d = float(np.sort(1.0 - fn @ gate["bank"].T, axis=1)[0, gate["k"] - 1])
    ok = d <= gate["radius"]
    ci = int(prob.argmax())
    return {
        "accepted": bool(ok),
        "species": gate["short"][ci] if ok else None,
        "confidence": float(prob[ci]),
        "distance": d,
        "radius": gate["radius"],
        "probs": prob.tolist(),
    }


def combine(results, n_models):
    """รวมคำตอบของหลายโมเดลให้เป็นคำตอบเดียวของปีกตัวนั้น

    กติกาเดียวกับ check_image.py:
      - ไม่มีโมเดลไหนผ่านเลย            -> ไม่รู้
      - ทุกโมเดลผ่านและตอบตรงกันหมด     -> ตอบชนิดนั้น
      - นอกนั้น ถ้าที่ผ่านตอบตรงกันหมด  -> ตอบชนิดนั้น (แต่ไม่ใช่ฉันทามติเต็ม)
      - ถ้าที่ผ่านตอบไม่ตรงกัน           -> ไม่รู้
    """
    good = [r["species"] for r in results if r["accepted"] and r["species"]]
    if not good:
        return None, "ทุกโมเดลปฏิเสธ"
    if len(set(good)) == 1 and len(good) == n_models:
        return good[0], "ทุกโมเดลตรงกัน"
    if len(set(good)) == 1:
        return good[0], f"{len(good)}/{n_models} โมเดลตอบ"
    return None, f"ที่ผ่านตอบไม่ตรงกัน ({', '.join(good)})"


# ====================================================== ทดสอบรูปเดียว
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("ใช้:  mlpy wing_detect.py \"path/to/image.jpg\"")
        raise SystemExit(1)

    path = sys.argv[1]
    archs = available_archs()
    if not archs:
        print(f"ไม่พบไฟล์โมเดลใน {model_dir()}")
        raise SystemExit(1)

    print(f"โฟลเดอร์โมเดล : {model_dir()}")
    print(f"อุปกรณ์        : {DEV}")
    gates = [load_gate(a) for a in archs]
    pf, fw, fh = geometry()
    print(f"โหลดแล้ว {len(gates)} โมเดล · pad_frac {pf} · frame {fw}x{fh}\n")

    arr = np.asarray(Image.open(path).convert("RGB"))
    H, W = arr.shape[:2]
    print("=" * 84)
    print(os.path.basename(path))
    print("=" * 84)
    print(f"  ขนาดภาพ {W}x{H} px")

    boxes, bg = find_wings(arr)
    if not boxes:
        print(f"  ตัวหาปีก : **หาปีกไม่เจอ** (พื้นหลัง {bg.mean():.0f})")
        print("  -> ปฏิเสธตั้งแต่ก่อนถึงโมเดล  ตอบ: ไม่รู้\n")
        raise SystemExit(0)

    print(f"  ตัวหาปีก : เจอ {len(boxes)} ตัว (พื้นหลัง {bg.mean():.0f})")
    answers = []
    for bi, box in enumerate(boxes, 1):
        bw, bh = box[2] - box[0] + 1, box[3] - box[1] + 1
        print(f"\n  ปีกที่ {bi}: ({box[0]},{box[1]})-({box[2]},{box[3]})  ขนาด {bw}x{bh} px")
        img = crop_for_model(arr, box)
        res = []
        for g in gates:
            r = predict_wing(g, img)
            res.append(r)
            mark = "ผ่าน " if r["accepted"] else "ปฏิเสธ"
            tail = (f"  -> {r['species']} ({r['confidence']*100:.0f}%)" if r["accepted"]
                    else f"  (เกินรัศมี {r['distance']-r['radius']:+.4f})")
            print(f"     {g['name']:<16} ระยะ {r['distance']:.4f} / "
                  f"รัศมี {r['radius']:.4f}   {mark}{tail}")
        sp, why = combine(res, len(gates))
        print(f"     สรุปปีกนี้: {'**' + sp + '**' if sp else '**ไม่รู้**'} — {why}")
        answers.append(sp)

    named = [a for a in answers if a]
    kinds = sorted(set(named))
    print("\n  == สรุปทั้งภาพ ==")
    print(f"  ปีกที่ตรวจเจอ {len(boxes)} ตัว · ตอบได้ {len(named)} ตัว · "
          f"ไม่รู้ {len(boxes)-len(named)} ตัว")
    print(f"  พบ {len(kinds)} ชนิด: {', '.join(kinds) if kinds else '(ไม่มี)'}\n")
