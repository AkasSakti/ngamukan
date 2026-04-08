# EEG Emotion Classification App

Streamlit app untuk klasifikasi emosi EEG menggunakan artifact model hasil training sebelumnya.

## Files

- `app.py`: aplikasi Streamlit
- `artifacts/`: model dan artifact preprocessing
- `requirements.txt`: dependency untuk local run dan Streamlit Cloud

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push repo ini ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository GitHub Anda.
4. Set `Main file path` ke `app.py`.
5. Deploy.

## Notes

- App tetap bisa jalan tanpa `dataset/emotions.csv` selama pengguna upload file CSV sendiri.
- Folder `artifacts/` harus ikut di repository karena dipakai langsung oleh `app.py`.
- Jika ukuran repository ingin diperkecil, dataset lokal dapat tidak di-push ke GitHub.
