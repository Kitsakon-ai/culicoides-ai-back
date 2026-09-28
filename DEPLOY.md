# คู่มือ deploy backend

FastAPI + โมเดลครอป 3 ตัว — ใช้ CPU อย่างเดียว ไม่ต้องใช้ GPU

---

## ไฟล์ที่ต้องมีบนโฮสต์

### ขั้นต่ำ (ใช้งานได้ครบทุกฟีเจอร์ ยกเว้นเทียบ 3 โมเดล) — **17.3 MB**

| ไฟล์ | ขนาด | หมายเหตุ |
|---|---|---|
| `main.py` | 36 KB | ตัวแอป (บางโฮสต์ต้องชื่อ `app.py` — Dockerfile รองรับทั้งสองชื่อ) |
| `wing_detect.py` | 20 KB | **ขาดไม่ได้** — ไม่มีแล้ว `/detect` ตอบ 503 ทั้งหมด |
| `requirements.txt` | — | |
| `Dockerfile` | — | เฉพาะโฮสต์ที่ใช้ Docker |
| `packages.txt` | — | เฉพาะ HF Space ที่ **ไม่ใช่** Docker SDK |
| `EfficientNetB0_crop_best.pth` | 15.6 MB | โมเดลเริ่มต้น |
| `knn_bank_effb0_crop.npz` | 1.6 MB | คลัง kNN ของโมเดลนั้น |

### เพิ่มถ้าอยากได้โหมด ensemble — รวมเป็น **138 MB**

| ไฟล์ | ขนาด |
|---|---|
| `DenseNet121_crop_best.pth` + `knn_bank_densenet121_crop.npz` | 28.4 MB |
| `ResNet50_crop_best.pth` + `knn_bank_resnet50_crop.npz` | 92.5 MB |

> **`.pth` กับ `.npz` ต้องมาเป็นคู่เสมอ** มีแต่ `.pth` ไม่มีคลัง = โมเดลนั้นโหลดไม่ขึ้น
> ระบบโหลดเท่าที่เจอไฟล์โดยอัตโนมัติ ขาดตัวไหนก็ข้ามตัวนั้น ไม่พัง
>
> ตัวเลขที่วัดไว้: ทั้ง 3 ตัวแม่น 100% เท่ากันบน test set 40 ภาพ
> **ResNet50 ไฟล์ใหญ่กว่า EfficientNet 5.8 เท่าโดยไม่ได้อะไรคืนมา** ถ้าพื้นที่จำกัดให้ตัดตัวนี้ก่อน

### ห้ามอัป

`.venv/` (198 MB และเป็นไบนารีของ Windows) · `__pycache__/` · `.git/` · `api_regression_detect.py` (สคริปต์ทดสอบ ไม่มีใคร import)

---

## ตามโฮสต์

### Hugging Face Space — Docker SDK

ต้องเลือก **CPU Basic** ก่อน ถึงจะเลือก Docker SDK ได้
(ZeroGPU รองรับเฉพาะ Gradio SDK — ถ้าเลือกไว้ ตัวเลือก Docker จะถูกล็อก)

`README.md` ต้องมี front-matter:

```yaml
---
title: Culicoides AI API
sdk: docker
app_port: 7860
---
```

### Hugging Face Space — Gradio SDK (กรณี Docker ใช้ไม่ได้)

ไม่ต้องใช้ `Dockerfile` แต่ต้องมี:

1. ตั้งชื่อไฟล์แอปเป็น **`app.py`** (HF สั่ง `python app.py`)
   — ท้ายไฟล์มี `if __name__ == "__main__"` ที่สั่ง uvicorn ให้แล้ว
2. **`packages.txt`** — HF ใช้ลง system package แทน `apt-get` ใน Dockerfile
3. เพิ่มบรรทัดแรกของ `requirements.txt` เพื่อไม่ให้ดึง torch บิลด์ CUDA ขนาด 2–3 GB:
   ```
   --extra-index-url https://download.pytorch.org/whl/cpu
   ```

```yaml
---
title: Culicoides AI API
sdk: gradio
app_file: app.py
---
```

### Render / Railway / Fly.io

ใช้ `Dockerfile` ได้เลย ไม่ต้องแก้อะไร — มันอ่าน `$PORT` ที่โฮสต์ส่งมาให้เอง

> Render free tier จะ **หลับเมื่อไม่มีคนใช้** ปลุกครั้งแรกช้า ~30–60 วินาที
> รวมกับเวลาโหลดโมเดล 3 ตัวอีก ~6 วินาที

### Google Cloud Run

ใช้ `Dockerfile` ได้เลย — บังคับว่าต้องฟัง `$PORT` ซึ่งรองรับแล้ว

```bash
gcloud run deploy culicoides-api --source . --region asia-southeast1 \
  --allow-unauthenticated --memory 2Gi
```

ต้องตั้ง `--memory 2Gi` ขึ้นไป เพราะโหลด 3 โมเดลพร้อมกัน

---

## หลัง deploy เสร็จ

### 1. ตรวจว่าขึ้นจริง

```
GET https://<host>/health
```

ต้องเห็น:

```json
"detect": { "ready": true, "models": [ {...}, {...}, {...} ] }
```

ถ้า `ready: false` ให้ดูช่อง `error` จะบอกว่าไฟล์ไหนขาด

### 2. ชี้ frontend มาที่ backend ใหม่

แก้ `culicoides-ai-front/.env`:

```
FASTAPI_URL='https://<host>'
```

แล้ว redeploy frontend

---

## ข้อควรระวังเรื่องพื้นที่

**Git เก็บทุกเวอร์ชันของไฟล์ไว้ตลอดกาล** — อัปโมเดล 94 MB ทับ 4 ครั้ง = กิน 376 MB ถึงจะเห็นไฟล์เดียวก็ตาม และลบใน commit ใหม่ **ไม่คืนพื้นที่**

HF Space free tier จำกัด **1 GB ต่อ repo** เคยเต็มมาแล้วด้วยสาเหตุนี้

**วิธีกัน:** อัปไฟล์โมเดลครั้งเดียวจบ ถ้าต้องแก้บ่อยให้แก้เฉพาะโค้ด

**ถ้าเต็มแล้ว:** บีบประวัติให้เหลือ commit เดียว (ย้อนกลับไม่ได้)

```bash
pip install -U huggingface_hub
hf auth login
python -c "from huggingface_hub import HfApi; HfApi().super_squash_history(repo_id='<user>/<space>', repo_type='space')"
```

---

## สิ่งที่แก้ไปแล้วใน Dockerfile (อย่าย้อนกลับ)

| จุด | เหตุผล |
|---|---|
| `libgl1` + `libglib2.0-0` | `python:3.10-slim` ไม่มีมาให้ → `import cv2` พังด้วย `libGL.so.1: cannot open shared object file` |
| `--index-url` **คู่กับ** `--extra-index-url` | ใส่แค่ตัวแรกจะแทนที่ PyPI ทั้งหมด แล้ว build ล้มที่ `flit_core` |
| `MPLCONFIGDIR=/tmp/matplotlib` | โฮสต์ที่ `$HOME` เขียนไม่ได้ matplotlib จะพังตอนสร้าง font cache |
| เลือกชื่อโมดูลอัตโนมัติ | `main.py` ในเครื่อง แต่ `app.py` บน HF — hardcode แล้วพังเงียบ ๆ |
| `${PORT:-7860}` | Cloud Run/Render กำหนดพอร์ตมาเอง ไม่ฟังตามจะถือว่า deploy ไม่สำเร็จ |
