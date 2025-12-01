import os
import google.generativeai as genai

try:
    # Pastikan Anda sudah mengatur environment variable GEMINI_API_KEY
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Environment variable GEMINI_API_KEY tidak ditemukan.")
        print("Silakan atur dengan perintah: set GEMINI_API_KEY=KUNCI_API_ANDA")
    else:
        genai.configure(api_key=api_key)
        print("Mencari model yang tersedia untuk kunci API Anda...")
        print("-" * 30)
        
        found_model = False
        for m in genai.list_models():
            # Memeriksa apakah model mendukung metode 'generateContent'
            if 'generateContent' in m.supported_generation_methods:
                print(f"Model Ditemukan: {m.name}")
                found_model = True
        
        if not found_model:
            print("\nTidak ada model yang mendukung 'generateContent' ditemukan untuk kunci API ini.")
        else:
            print("-" * 30)
            print("\nSilakan coba gunakan salah satu nama model di atas dalam file app.py.")

except Exception as e:
    print(f"\nTerjadi kesalahan saat mencoba menghubungi Google AI: {e}")
    print("Pastikan kunci API Anda valid dan Anda memiliki koneksi internet.")

