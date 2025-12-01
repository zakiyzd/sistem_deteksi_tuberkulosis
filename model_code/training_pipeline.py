# =============================================================================
# 1. IMPORT PUSTAKA
# =============================================================================
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

print(f"TensorFlow Version: {tf.__version__}")

# =============================================================================
# 2. DEFINISI JALUR & PARAMETER
# =============================================================================
# --- Jalur File ---
# Menggunakan os.path.join untuk path yang lebih robust
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXCEL_PATH_TB = os.path.join(BASE_DIR, 'data', 'metadata_tuberkulosis.xlsx')
EXCEL_PATH_NORMAL = os.path.join(BASE_DIR, 'data', 'metadata_normal.xlsx')
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'all_image')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'web_app', 'tb_resnet_best.h5')

# --- Nama Kolom Excel ---
NAMA_KOLOM_FILE = 'FILE NAME'

# --- Parameter Training ---
TARGET_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 30

# =============================================================================
# 3. MEMUAT DAN MEMPROSES DATA (dari Notebook 01)
# =============================================================================
print("\n--- Langkah 3: Memuat Data ---")
# Membaca data TB
df_tb = pd.read_excel(EXCEL_PATH_TB)
df_tb.rename(columns={NAMA_KOLOM_FILE: 'File_Name_Clean'}, inplace=True)
df_tb['Label'] = 'Tuberculosis'

# Membaca data Normal
df_normal = pd.read_excel(EXCEL_PATH_NORMAL)
df_normal.rename(columns={NAMA_KOLOM_FILE: 'File_Name_Clean'}, inplace=True)
df_normal['Label'] = 'Normal'

# Gabungkan
df = pd.concat([df_tb, df_normal], ignore_index=True)

# Buat path lengkap ke gambar
df['path'] = df['File_Name_Clean'].astype(str).str.strip() + '.png'
df['path'] = df['path'].apply(lambda x: os.path.join(IMAGE_DIR, x))

print(f"Total data gabungan: {len(df)} baris")
print("Distribusi Kelas Awal:")
print(df['Label'].value_counts())

# =============================================================================
# 4. PEMBAGIAN DATA (dari Notebook 01)
# =============================================================================
print("\n--- Langkah 4: Membagi Data ---")
df_train, df_temp = train_test_split(
    df, test_size=0.30, stratify=df['Label'], random_state=42
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, stratify=df_temp['Label'], random_state=42
)

print(f"Total Data Latih: {len(df_train)}")
print(f"Total Data Validasi: {len(df_val)}")
print(f"Total Data Uji: {len(df_test)}")

# =============================================================================
# 5. PERHITUNGAN CLASS WEIGHTS (dari Notebook 01)
# =============================================================================
print("\n--- Langkah 5: Menghitung Bobot Kelas ---")
nama_kelas_unik = np.sort(df['Label'].unique())
bobot_kelas_array = compute_class_weight(
    class_weight='balanced', classes=nama_kelas_unik, y=df_train['Label']
)
class_indices = {name: i for i, name in enumerate(nama_kelas_unik)}
bobot_kelas_dictionary = {
    class_indices[name]: bobot_kelas_array[i] for i, name in enumerate(nama_kelas_unik)
}
print("Bobot Kelas yang akan digunakan:", bobot_kelas_dictionary)

# =============================================================================
# 6. SETUP DATA GENERATOR (dari Notebook 01)
# =============================================================================
print("\n--- Langkah 6: Menyiapkan Data Generator ---")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col='path',
    y_col='Label',
    target_size=TARGET_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col='path',
    y_col='Label',
    target_size=TARGET_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col='path',
    y_col='Label',
    target_size=TARGET_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =============================================================================
# 7. DEFINISI & KOMPILASI MODEL (dari Notebook 02)
# =============================================================================
print("\n--- Langkah 7: Membangun Model ---")
base_model = ResNet50(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# =============================================================================
# 8. PELATIHAN MODEL (dari Notebook 02)
# =============================================================================
print("\n--- Langkah 8: Memulai Pelatihan Model ---")
checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    class_weight=bobot_kelas_dictionary, # INI BAGIAN PENTINGNYA
    callbacks=[checkpoint]
)

print("\n--- Pelatihan Selesai ---")
print(f"Model terbaik dari tahap awal telah disimpan di: {MODEL_SAVE_PATH}")

# =============================================================================
# 9. FINE-TUNING MODEL
# =============================================================================
print("\n--- Langkah 9: Memulai Fine-Tuning ---")

# Unfreeze some layers of the base model
base_model.trainable = True
fine_tune_at = 143  # Unfreeze from this layer onwards (a common choice for ResNet50)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Re-compile the model with a very low learning rate
optimizer_fine_tune = Adam(learning_rate=1e-5)
model.compile(
    optimizer=optimizer_fine_tune,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

print("Model re-compiled for fine-tuning.")

# Continue training
fine_tune_epochs = 20
total_epochs = EPOCHS + fine_tune_epochs

history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    class_weight=bobot_kelas_dictionary,
    callbacks=[checkpoint]  # Use the same checkpoint to save the best model
)

print("\n--- Fine-tuning Selesai ---")
print(f"Model terbaik setelah fine-tuning telah disimpan di: {MODEL_SAVE_PATH}")

# =============================================================================
# 10. EVALUASI MODEL SETELAH FINE-TUNING
# =============================================================================
print("\n--- Langkah 10: Mengevaluasi model setelah fine-tuning dengan data uji ---")
# Muat model terbaik yang disimpan
model.load_weights(MODEL_SAVE_PATH)

results = model.evaluate(test_generator)

# --- Cetak hasil ke konsol ---
print("\nHasil Evaluasi Data Uji (Setelah Fine-Tuning):")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")

# --- Simpan hasil ke file ---
EVAL_RESULT_PATH = os.path.join(os.path.dirname(__file__), 'evaluation_results.txt')
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(EVAL_RESULT_PATH, "a") as f:
    f.write(f"--- Hasil Evaluasi pada {timestamp} ---\n")
    f.write(f"Loss: {results[0]:.4f}\n")
    f.write(f"Accuracy: {results[1]:.4f}\n")
    f.write(f"Precision: {results[2]:.4f}\n")
    f.write(f"Recall: {results[3]:.4f}\n")
    f.write("="*40 + "\n")

print(f"\nHasil evaluasi juga telah disimpan di: {EVAL_RESULT_PATH}")
