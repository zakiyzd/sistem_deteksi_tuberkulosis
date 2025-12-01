from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# Model Tabel Pengguna
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False) # Simpan password tanpa hashing untuk skripsi sederhana
    role = db.Column(db.String(50), nullable=False, default='Klinis')
    predictions = db.relationship('PredictionHistory', backref='author', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

# Tabel Riwayat Prediksi
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(100), nullable=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(20), nullable=False)
    patient_address = db.Column(db.String(200), nullable=True)
    prediction_result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<PredictionHistory {self.id}>'