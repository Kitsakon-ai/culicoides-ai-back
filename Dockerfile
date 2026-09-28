FROM python:3.10-slim

# opencv-python ต้องการ libGL กับ libglib — บน python:3.10-slim ไม่มีมาให้
# ถ้าไม่ลง จะพังตอน import cv2 ด้วย "libGL.so.1: cannot open shared object file"
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HF Space รันด้วยผู้ใช้ที่ไม่ใช่ root และ $HOME เขียนไม่ได้
# ถ้าไม่ชี้ cache มาที่ /tmp matplotlib จะพังตอนสร้าง font cache ครั้งแรก
ENV MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

# torch จาก PyPI เป็นบิลด์ CUDA ขนาด ~2-3 GB แต่ Space นี้รันบน CPU
# ดึงบิลด์ CPU ก่อน แล้ว requirements.txt จะเห็นว่ามีแล้วและข้ามไป
#
# ต้องใส่ทั้งสอง index:
#   --index-url       = index ของ PyTorch (มีเฉพาะ torch และของที่เกี่ยวข้อง)
#   --extra-index-url = PyPI สำหรับ dependency ที่เหลือ (filelock, sympy, flit_core ฯลฯ)
# ถ้าใส่แค่ --index-url ตัวแรก pip จะหา dependency พวกนั้นจาก index ของ PyTorch
# แล้วล้มด้วย "Could not find a version that satisfies the requirement flit_core"
#
# เวอร์ชัน +cpu ถือเป็น local version ซึ่งเรียงสูงกว่าเวอร์ชันเปล่าบน PyPI
# pip จึงเลือกบิลด์ CPU ให้เองโดยไม่ต้องระบุเวอร์ชัน
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch torchvision \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# รันได้ทุกโฮสต์โดยไม่ต้องแก้ไฟล์ — จัดการ 2 เรื่องที่ต่างกันในแต่ละที่:
#
#   1. ชื่อไฟล์  ในเครื่อง/GitHub = main.py  แต่บน HF Space = app.py
#      เลือกให้อัตโนมัติจากไฟล์ที่มีอยู่จริง แทนที่จะ hardcode แล้วพังเงียบ ๆ
#
#   2. พอร์ต     HF Space = 7860 · Render/Railway/Cloud Run ส่ง $PORT มาให้
#      ถ้าไม่ฟังพอร์ตที่โฮสต์กำหนด Cloud Run จะถือว่า deploy ไม่สำเร็จ
#
# ใช้ exec เพื่อให้ uvicorn เป็น PID 1 — สัญญาณ SIGTERM ตอนปิดคอนเทนเนอร์
# จะส่งถึงตัว uvicorn โดยตรง ปิดงานค้างได้สะอาด ไม่โดน kill ทิ้ง
CMD ["sh", "-c", "exec uvicorn $(test -f main.py && echo main || echo app):app --host 0.0.0.0 --port ${PORT:-7860}"]
