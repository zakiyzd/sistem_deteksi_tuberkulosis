import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, abort, make_response, Response, send_file, after_this_request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import io
import pandas as pd
from datetime import datetime
from pytz import timezone
from fpdf import FPDF
# cv2, kmodels TIDAK digunakan karena Grad-CAM sudah dihapus
from flask_login import LoginManager, login_user, logout_user, login_required, current_user 

# Import model dan form lokal
from models import db, User, PredictionHistory
from forms import RegistrationForm, LoginForm


# --- 0. Konfigurasi Awal dan Setup Database ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'kunci_rahasia_untuk_skripsi_sangat_aman'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inisialisasi Pustaka
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 


# --- 1. Pemuatan Model Global ---
MODEL_PATH = 'tb_resnet_best.h5' 
TARGET_SIZE = (224, 224) 
CLASSES = ['Normal', 'Tuberculosis'] 


try:
    model = load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model ResNet50 berhasil dimuat dan siap digunakan.")
    
except Exception as e:
    print(f"ERROR: Gagal memuat model. Detail Error: {e}")
    model = None


# --- 2. Fungsi Login Manager (Wajib) ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- 3. Fungsi Pra-pemrosesan Citra ---
def preprocess_image(img_stream):
    if img_stream is None:
        return None
    img = Image.open(io.BytesIO(img_stream)).convert('RGB')
    img = img.resize(TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array


# --- 4. Route Pendaftaran (Register) ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, password=form.password.data, role='Klinis')
        try:
            db.session.add(user)
            db.session.commit()
            flash('Akun berhasil dibuat! Silakan Login.', 'success')
            return redirect(url_for('login'))
        except Exception:
            flash('Gagal membuat akun. Username sudah terdaftar.', 'error')
            db.session.rollback()

    return render_template('register.html', form=form)


# --- 4a. Route Halaman Utama (/) ---
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


# --- 4b. Route Halaman Tentang & Kontak ---
@app.route('/about-contact')
def about_contact():
    return render_template('about_contact.html')


# --- 5. Route Login (/login) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user and user.password == form.password.data:
            login_user(user)
            flash(f"Selamat datang, {user.username}!", "success")
            return redirect(url_for('home'))
        else:
            flash("Username atau Password salah.", "error")
            
    return render_template('login.html', form=form)

# --- 6. Route Deteksi (/deteksi) ---
@app.route('/deteksi')
@login_required
def deteksi():
    # Ambil riwayat prediksi untuk pengguna yang sedang login, urutkan dari yang terbaru
    history_utc = PredictionHistory.query.filter_by(author=current_user).order_by(PredictionHistory.timestamp.desc()).all()
    
    # Tentukan zona waktu
    utc = timezone('UTC')
    wib = timezone('Asia/Jakarta')
    
    # Konversi timestamp
    history_wib = []
    for item in history_utc:
        # Tambahkan informasi timezone UTC ke timestamp dari DB
        utc_timestamp = utc.localize(item.timestamp)
        # Konversi ke WIB
        wib_timestamp = utc_timestamp.astimezone(wib)
        item.timestamp = wib_timestamp
        history_wib.append(item)

    return render_template('index.html', username=current_user.username, history=history_wib)

# --- 7. Route Logout (/logout) ---
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Anda berhasil logout.", "info")
    return redirect(url_for('login'))


# --- 8. Route API: Prediksi (/predict) ---
@app.route('/predict', methods=['POST'])
@login_required 
def predict():
    if model is None:
        return jsonify({'error': 'Model tidak dimuat.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file citra yang diterima.'}), 400
    
    file = request.files['file']
    img_stream = file.read()
    processed_image = preprocess_image(img_stream)

    if processed_image is None:
        return jsonify({'error': 'Gagal memproses file citra.'}), 500

    patient_name = request.form.get('patient_name')
    patient_age = request.form.get('patient_age')
    patient_id = request.form.get('patient_id')
    patient_gender = request.form.get('patient_gender')
    patient_address = request.form.get('patient_address')

    try:
        _ = model(processed_image, training=False) 
    except:
        pass 
    
    raw_prediction = model.predict(processed_image)
    prob_tb = raw_prediction[0][0] 
    
    if prob_tb >= 0.5:
        predicted_class = CLASSES[1] # Tuberculosis
    else:
        predicted_class = CLASSES[0] # Normal
    
    confidence = max(prob_tb, 1 - prob_tb) * 100
    
    original_filename = f"{current_user.username}_{os.path.basename(file.filename)}"
    image_save_path = os.path.join(app.root_path, 'static', 'original_' + original_filename)
    
    with open(image_save_path, 'wb') as f:
        f.write(img_stream)
    
    image_url_for_db = f'/static/original_{original_filename}'

    # Simpan riwayat ke database
    try:
        valid_age = 0
        if patient_age and patient_age.isdigit():
            valid_age = int(patient_age)

        new_history = PredictionHistory(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_age=valid_age,
            patient_gender=patient_gender,
            patient_address=patient_address,
            prediction_result=predicted_class,
            confidence=f'{confidence:.2f}%',
            image_path=image_url_for_db,
            author=current_user 
        )
        db.session.add(new_history)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"--- DATABASE ERROR: Gagal menyimpan riwayat prediksi. ---")
        print(f"--- Error Detail: {e} ---")

    try:
        # Convert timestamp to WIB for the response
        utc = timezone('UTC')
        wib = timezone('Asia/Jakarta')
        utc_timestamp = utc.localize(new_history.timestamp)
        wib_timestamp = utc_timestamp.astimezone(wib)
        
        new_history_data = {
            'id': new_history.id,
            'timestamp': wib_timestamp.strftime('%Y-%m-%d %H:%M'),
            'patient_id': new_history.patient_id,
            'patient_name': new_history.patient_name,
            'patient_age_gender': f'{new_history.patient_age} thn / {new_history.patient_gender}',
            'prediction_result': new_history.prediction_result,
            'confidence': new_history.confidence,
            'image_path': new_history.image_path
        }
    except Exception as e:
        print(f"--- TIMESTAMP/DATA PREP ERROR: {e} ---")
        new_history_data = None

    result = {
        'prediction': predicted_class,
        'confidence': f'{confidence:.2f}%',
        'image_url': image_url_for_db,
        'patient_info': {
            'id': patient_id,
            'name': patient_name,
            'age': patient_age,
            'gender': patient_gender,
            'address': patient_address
        },
        'new_history': new_history_data
    }
    
    return jsonify(result)

# --- 9. Route Hapus Riwayat ---
@app.route('/delete/<int:history_id>', methods=['POST'])
@login_required
def delete_history(history_id):
    history_item = PredictionHistory.query.get_or_404(history_id)
    if history_item.author != current_user:
        abort(403) # Forbidden
    try:
        db.session.delete(history_item)
        db.session.commit()
        flash('Riwayat diagnosis telah berhasil dihapus.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Gagal menghapus riwayat. Error: {e}', 'error')
    return redirect(url_for('deteksi'))

@app.route('/edit/<int:history_id>', methods=['GET', 'POST'])
@login_required
def edit_history(history_id):
    history_item = PredictionHistory.query.get_or_404(history_id)
    if history_item.author != current_user:
        abort(403)

    if request.method == 'POST':
        try:
            history_item.patient_name = request.form['patient_name']
            history_item.patient_id = request.form['patient_id']
            history_item.patient_age = int(request.form['patient_age'])
            history_item.patient_gender = request.form['patient_gender']
            history_item.patient_address = request.form['patient_address']
            
            db.session.commit()
            flash('Data riwayat berhasil diperbarui.', 'success')
            return redirect(url_for('deteksi'))
        except Exception as e:
            db.session.rollback()
            flash(f'Gagal memperbarui data. Error: {e}', 'error')
            
    return render_template('edit_history.html', item=history_item)

@app.route('/download_pdf/<int:history_id>')
@login_required
def download_pdf(history_id):
    history_item = PredictionHistory.query.get_or_404(history_id)
    if history_item.author != current_user:
        abort(403)

    pdf = FPDF()
    pdf.add_page()

    # Judul
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Laporan Hasil Diagnosis TB', 0, 1, 'C')
    pdf.ln(10)

    # Data Pasien
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Data Pasien', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(40, 8, f'Nama Pasien:')
    pdf.cell(0, 8, f'{history_item.patient_name}', 0, 1)
    pdf.cell(40, 8, f'ID Pasien/No. RM:')
    pdf.cell(0, 8, f'{history_item.patient_id}', 0, 1)
    pdf.cell(40, 8, f'Umur:')
    pdf.cell(0, 8, f'{history_item.patient_age} tahun', 0, 1)
    pdf.cell(40, 8, f'Jenis Kelamin:')
    pdf.cell(0, 8, f'{history_item.patient_gender}', 0, 1)
    pdf.cell(40, 8, f'Alamat:')
    pdf.multi_cell(0, 8, f'{history_item.patient_address}')
    pdf.ln(10)

    # Hasil Diagnosis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Hasil Diagnosis Berbasis AI', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(40, 8, 'Hasil Deteksi:')
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, f'{history_item.prediction_result}', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(40, 8, 'Tingkat Keyakinan:')
    pdf.cell(0, 8, f'{history_item.confidence}', 0, 1)
    pdf.ln(10)

    # Deskripsi dan Saran
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Deskripsi dan Saran Tindak Lanjut', 0, 1)
    pdf.set_font('Arial', '', 11)
    if history_item.prediction_result == 'Tuberculosis':
        pdf.multi_cell(0, 8, 'Hasil analisis citra menunjukkan adanya indikasi kuat Tuberkulosis (TB). Diperlukan evaluasi medis segera untuk konfirmasi dan penanganan lebih lanjut.')
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Saran:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, '1. Segera konsultasikan hasil ini dengan dokter atau spesialis paru.\n2. Lakukan pemeriksaan lanjutan seperti tes dahak (BTA) atau tes Mantoux sesuai anjuran dokter.\n3. Hindari kontak dekat dengan orang lain, selalu gunakan masker, dan praktikkan etika batuk untuk mencegah penularan.')
    else:
        pdf.multi_cell(0, 8, 'Hasil analisis citra menunjukkan kondisi paru-paru tampak Normal. Tidak ditemukan adanya indikasi Tuberkulosis (TB) pada citra X-ray.')
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Saran:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 8, '1. Tetap jaga pola hidup sehat dengan tidak merokok dan menghindari asap rokok.\n2. Lakukan olahraga secara teratur untuk menjaga kesehatan paru-paru.\n3. Jika mengalami gejala gangguan pernapasan (batuk lama, sesak napas), segera periksakan diri ke dokter.')
    pdf.ln(15)

    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 10, 'Laporan ini dibuat secara otomatis oleh sistem deteksi TB berbasis AI dan bukan merupakan diagnosis klinis final.', 0, 1, 'C')

    # Output PDF
    temp_pdf_path = os.path.join(app.instance_path, f'temp_report_{history_item.id}.pdf')
    pdf.output(temp_pdf_path)

    @after_this_request
    def remove_file(response):
        try:
            os.remove(temp_pdf_path)
        except Exception as e:
            app.logger.error(f"Error removing temporary file: {e}")
        return response

    return send_file(temp_pdf_path, as_attachment=True, download_name=f'Laporan_{history_item.patient_name}.pdf')

@app.route('/download_excel')
@login_required
def download_excel():
    history_items = PredictionHistory.query.filter_by(author=current_user).order_by(PredictionHistory.timestamp.desc()).all()

    if not history_items:
        flash('Tidak ada riwayat diagnosis untuk diunduh.', 'info')
        return redirect(url_for('deteksi'))

    # Konversi zona waktu
    utc = timezone('UTC')
    wib = timezone('Asia/Jakarta')
    
    records = []
    for item in history_items:
        utc_timestamp = utc.localize(item.timestamp)
        wib_timestamp = utc_timestamp.astimezone(wib)
        records.append({
            'Waktu Diagnosis': wib_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'ID Pasien/No. RM': item.patient_id,
            'Nama Pasien': item.patient_name,
            'Umur': item.patient_age,
            'Jenis Kelamin': item.patient_gender,
            'Alamat': item.patient_address,
            'Hasil Diagnosis AI': item.prediction_result,
            'Tingkat Keyakinan': item.confidence
        })

    df = pd.DataFrame(records)

    # Buat file Excel di memori
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Riwayat Diagnosis')
    
    # Atur lebar kolom otomatis
    worksheet = writer.sheets['Riwayat Diagnosis']
    for i, col in enumerate(df.columns):
        column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
        worksheet.set_column(i, i, column_len)

    writer.close()
    output.seek(0)

    # Buat nama file dinamis
    timestamp_str = datetime.now(wib).strftime('%Y%m%d_%H%M%S')
    filename = f"Laporan_Riwayat_Diagnosis_{current_user.username}_{timestamp_str}.xlsx"

    return Response(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment;filename={filename}'}
    )

# --- 10. Menjalankan Aplikasi dan Setup DB ---

# --- Konfigurasi Gemini API ---
# Ganti dengan API key Anda. Sebaiknya gunakan environment variable.
# os.environ['GEMINI_API_KEY'] = "YOUR_API_KEY"
try:
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('models/gemini-flash-latest')
    print("Model Gemini (models/gemini-flash-latest) berhasil dikonfigurasi.")
except Exception as e:
    gemini_model = None
    print(f"ERROR: Gagal mengkonfigurasi Gemini. Pastikan Anda telah mengatur GEMINI_API_KEY. Detail Error: {e}")


@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    if not gemini_model:
        return jsonify({'response': 'Konfigurasi model AI (Gemini) gagal. Pastikan Anda telah mengatur environment variable GEMINI_API_KEY dengan benar.'}), 200

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Pesan tidak ditemukan.'}), 400

    user_message = data['message']

    try:
        # Menambahkan konteks atau instruksi khusus untuk model
        prompt = f"Anda adalah asisten AI untuk aplikasi deteksi Tuberkulosis. Jawab pertanyaan pengguna seputar TB, kesehatan paru-paru, atau cara penggunaan aplikasi. Jaga agar jawaban tetap informatif, singkat, dan ramah. Pertanyaan pengguna: '{user_message}'"
        response = gemini_model.generate_content(prompt)
        
        # Periksa apakah ada teks dalam respons dan tidak diblokir
        if not response.parts:
            if response.prompt_feedback.block_reason:
                bot_response = f"Permintaan Anda diblokir karena: {response.prompt_feedback.block_reason.name}. Silakan ubah pertanyaan Anda."
            else:
                bot_response = "Maaf, saya tidak dapat memberikan respons saat ini. Silakan coba lagi."
        else:
            bot_response = response.text

    except Exception as e:
        print(f"Error saat memanggil Gemini API: {e}")
        # Memberikan pesan error yang lebih spesifik berdasarkan jenis exception
        if 'API_KEY' in str(e).upper() or 'permission' in str(e).lower() or 'authentication' in str(e).lower():
             bot_response = "Terjadi kesalahan otentikasi dengan API model AI. Periksa kembali GEMINI_API_KEY Anda."
        else:
            bot_response = "Maaf, terjadi kesalahan pada server AI. Silakan coba lagi nanti."

    return jsonify({'response': bot_response})


if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
        
        if User.query.filter_by(username='admin').first() is None:
            admin_user = User(username='admin', password='password', role='Admin')
            db.session.add(admin_user)
            db.session.commit()
            print("Akun 'admin' default (pass: password) telah dibuat.")
            
        print("Database terhubung dan tabel pengguna siap.")

    os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)
    app.run(debug=True)