# Vietnamese Subtitle OCR

Tự động trích xuất và làm sạch phụ đề tiếng Việt từ video YouTube bằng OCR.

## 📋 Quy trình

```
YouTube Video
    ↓
[Step 1] Download Video
    ↓
[Step 2] Extract Subtitles (Surya OCR)
    ↓
[Step 3] Clean & Fix (Gemini AI)
    ↓
Cleaned .SRT File
```

## 🚀 Setup

### 1. Clone Repository
```bash
git clone https://github.com/trongnghiapix/vid_ocr.git
cd vid_ocr
```

### 2. Tạo Virtual Environment
```bash
python3 -m venv .venv

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Hoặc Windows
.venv\Scripts\activate
```

### 3. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup File `.env`

Tạo file `.env` ở thư mục gốc với nội dung:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

#### Cách lấy Gemini API Key:
1. Truy cập [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy key và paste vào file `.env`

**⚠️ Lưu ý:** Không commit file `.env` lên git (đã được ignore)

## 💻 Cách Sử Dụng

### Step 1: Download Video từ YouTube
```bash
python download_video.py "https://www.youtube.com/watch?v=..."

# Tùy chọn:
# python download_video.py "URL" --output-dir downloads --name vid
```
Output: `downloads/vid.mp4`

### Step 2: Trích Xuất Phụ Đề (OCR)
```bash
python extract_subtitles.py downloads/vid.mp4

# Tùy chọn:
# python extract_subtitles.py downloads/vid.mp4 --ocr surya --fps 2
```
Output: `downloads/vid.srt`

### Step 3: Làm Sạch Phụ Đề (AI)
```bash
python clean_srt.py downloads/vid.srt

# Tùy chọn:
# python clean_srt.py downloads/vid.srt --model gemini-2.0-flash
```
Output: `downloads/vid.cleaned.srt`

## 🛠️ Công Cụ Sử Dụng

- **yt-dlp** — Tải video từ YouTube
- **Surya OCR** — Nhận dạng ký tự từ hình ảnh
- **Gemini AI** — Sửa lỗi chính tả và làm sạch phụ đề
- **OpenCV** — Xử lý video

## 📁 Cấu Trúc Project

```
vid_ocr/
├── download_video.py       # Step 1: Tải video
├── extract_subtitles.py    # Step 2: Trích xuất phụ đề
├── clean_srt.py            # Step 3: Làm sạch phụ đề
├── backends/
│   └── surya.py            # OCR backend
├── sample/                 # Video mẫu
├── downloads/              # Thư mục lưu output
├── requirements.txt        # Dependencies
├── .env                    # API keys (không commit)
└── README.md               # Tài liệu này
```

## ❓ Troubleshooting

### "Device not configured" khi push git
→ Sử dụng Personal Access Token (PAT) thay vì password

### OCR chậm trên Mac
→ Adjust `SAMPLE_FPS` nhỏ hơn trong `extract_subtitles.py`

### Gemini API key không hoạt động
→ Kiểm tra file `.env` được tạo đúng cách và API key hợp lệ

## 📝 License

MIT

## 👨‍💻 Author

trongnghiapix
