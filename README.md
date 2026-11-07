# Real-Time Motorcycle Accident Detection and Telegram Notification System with Low-Light Enhancement and ByteTrack

ระบบตรวจจับอุบัติเหตุรถจักรยานยนต์แบบเรียลไทม์และแจ้งเตือนผ่าน Telegram ร่วมกับการปรับปรุงภาพแสงน้อย (CIE LAB + Adaptive Gamma + CLAHE) และการติดตามวัตถุด้วย ByteTrack

## สถาปัตยกรรมระบบ (System Architecture)
1. **Adaptive Low-Light Enhancement (`preprocessing/low_light.py`)**: แปลงภาพจาก BGR เป็น CIE LAB เพื่อตรวจวัดค่าความสว่างเฉลี่ยบนช่อง $L$-channel หากต่ำกว่าค่าที่กำหนด ($L < 90$) ระบบจะทำ Adaptive Gamma Correction ร่วมกับ CLAHE เพื่อเพิ่มรายละเอียดในที่มืดโดยไม่ทำให้สีเพี้ยน
2. **YOLO11 + ByteTrack Multi-Object Tracking (`inference.py`)**: ตรวจจับและติดตามผู้ประสบอุบัติเหตุ (`person_accident`) และรถจักรยานยนต์ที่เกิดอุบัติเหตุ (`motorcycle_accident`) พร้อมระบบยืนยันผลต่อเนื่องหลายเฟรม (Multi-frame Temporal Confirmation 5 เฟรม) เพื่อลดการแจ้งเตือนผิดพลาด (False Positive)
3. **Asynchronous Telegram Notification (`notification/telegram_bot.py`)**: ส่งภาพหลักฐานพร้อมข้อความ HTML สรุปเวลา, จำนวนรถจักรยานยนต์, จำนวนผู้บาดเจ็บ และค่าความเชื่อมั่น (Confidence Score) แบบแยก Thread โดยไม่บล็อกการทำงานของระบบประมวลผลภาพหลัก
4. **Academic Visualization & CSV Telemetry (`utils/visualizer.py`, `utils/logger.py`, `evaluate_ablation.py`)**: แสดงผลกรอบสีแดง (`BGR: 0, 0, 255`) สำหรับผู้บาดเจ็บ และกรอบสีเหลือง (`BGR: 0, 255, 255`) สำหรับรถจักรยานยนต์ที่เกิดอุบัติเหตุ พร้อมบันทึกค่า FPS, Latency และผลลัพธ์ลงไฟล์ CSV สำหรับใช้วิเคราะห์ในงานวิจัย

## โครงสร้างโปรเจกต์ (Project Structure)
```text
├── config.yaml                  # ไฟล์ตั้งค่าโมเดล, Low-Light, Temporal Confirmation และ Telegram Bot
├── requirements.txt             # รายการไลบรารีที่ต้องติดตั้ง
├── preprocessing/
│   └── low_light.py             # คลาส AdaptiveLowLightEnhancer (LAB + Adaptive Gamma + CLAHE)
├── notification/
│   └── telegram_bot.py          # คลาส TelegramNotifier (Asynchronous Threading + HTML Caption)
├── utils/
│   ├── visualizer.py            # คลาส AccidentVisualizer (Red/Yellow Bounding Boxes + HUD)
│   └── logger.py                # คลาส ResearchCSVLogger บันทึกสถิติ FPS/Latency และเหตุการณ์อุบัติเหตุ
├── inference.py                 # ไปป์ไลน์หลักสำหรับรันตรวจจับแบบเรียลไทม์
└── evaluate_ablation.py         # สคริปต์รันเปรียบเทียบผลการทดลอง (Ablation Study 4 รูปแบบ)
```

## วิธีการติดตั้งและใช้งาน (Quick Start)

```bash
# 1. สร้าง Virtual Environment และติดตั้งไลบรารี
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# 2. รันระบบตรวจจับอุบัติเหตุ
./.venv/bin/python inference.py --config config.yaml --source path/to/video.mp4

# 3. รันประเมินผล Ablation Study
./.venv/bin/python evaluate_ablation.py --config config.yaml --output-csv logs/ablation_results.csv --synthetic
```
