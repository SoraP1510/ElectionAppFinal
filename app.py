import pandas as pd
from PIL import Image
import numpy as np
import json
from pyngrok import ngrok
import tensorflow as tf
import shutil
import random
import cv2
import os
import time
from dotenv import load_dotenv
import base64
from flask import Flask, request, render_template, make_response, redirect, url_for, send_from_directory, session, jsonify
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# CONFIGURATION
load_dotenv()
base_path = './Face_Recog_App'
user_img_dir = f'{base_path}/static/uploads/UserImages'
os.makedirs(user_img_dir, exist_ok=True)

csv_file = f'{user_img_dir}/users.csv'
vote_file = f'{user_img_dir}/votes.csv' 

# รหัสผ่านสำหรับเข้า Admin Zone
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# โหลด Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['name', 'surname', 'phone', 'folder', 'has_voted'])
    df.to_csv(csv_file, index=False)

if not os.path.exists(vote_file):
    df_votes = pd.DataFrame(columns=['candidate_id', 'candidate_name', 'vote_count'])
    df_votes.to_csv(vote_file, index=False)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# HELPER FUNCTIONS

def cv2_imread_utf8(path):
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR) 
        stream.close()
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def detect_and_crop_face(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0: return None
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = image_array[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (100, 100))
    return face_roi

def load_data_rgb(data_path, img_size=(100,100)):
    images = []
    labels = []
    label_map = {}
    label_id = 0
    print(f"กำลังสแกนโฟลเดอร์: {data_path}")
    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path): continue
        if folder not in label_map:
            label_map[folder] = label_id
            label_id += 1     
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2_imread_utf8(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_map[folder])

    if len(images) > 0:
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
    else:
        images = np.empty((0, 100, 100, 3))
        labels = np.array([])
    return images, labels, label_map

def process_upload_to_cv2(file_storage):
    in_memory_file = file_storage.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# FLASK ROUTES

@app.route('/register', methods=['GET'])
def register_get():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    try:
        print("\n=== เริ่มกระบวนการลงทะเบียน ===")
        name = request.form['name']
        surname = request.form['surname']
        phone = request.form['phone']

        df = pd.read_csv(csv_file, dtype={'phone': str})
        if phone in df['phone'].values:
            return jsonify({'status': 'error', 'message': "เบอร์โทรศัพท์นี้ลงทะเบียนไปแล้ว"}), 400

        user_folder = f"{name}_{surname}".replace(" ", "_")
        save_path = os.path.join(user_img_dir, user_folder)
        os.makedirs(save_path, exist_ok=True)

        valid_images_count = 0
        example_image_b64 = None  # ตัวแปรเก็บรูปตัวอย่างที่มีกรอบ

        for i in range(8):
            file = request.files.get(f'image_{i}')
            if file:
                img = process_upload_to_cv2(file)
                if img is not None:
                    # จับหน้า
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                    if len(faces) > 0:
                        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                        
                        # 1. ส่วนบันทึกลง Disk
                        face_roi = img[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (100, 100))
                        cv2.imwrite(os.path.join(save_path, f'img_{i+1}.jpg'), face_roi)
                        valid_images_count += 1

                        # 2. ส่วนสร้างภาพตัวอย่าง
                        if example_image_b64 is None:
                            debug_img = img.copy()
                            # วาดกรอบสีเขียวลงในรูป copy
                            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            _, buffer = cv2.imencode('.jpg', debug_img)
                            example_image_b64 = base64.b64encode(buffer).decode('utf-8')

        if valid_images_count == 0:
            return jsonify({'status': 'error', 'message': "ไม่พบใบหน้าในรูปภาพที่ส่งมา กรุณาถ่ายใหม่ให้ชัดเจน"}), 400
        
        new_row = pd.DataFrame([[name, surname, phone, save_path, 0]], 
                            columns=['name', 'surname', 'phone', 'folder', 'has_voted'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f"✅ บันทึกข้อมูล {name} สำเร็จ ({valid_images_count} รูป)")

        print("⏳ กำลังเตรียมข้อมูลสำหรับเทรนโมเดล...")
        all_images, all_labels, label_map = load_data_rgb(user_img_dir)
        
        if len(all_images) > 0:
            num_classes = len(label_map)
            if num_classes > 1:
                trainX, testX, trainY, testY = train_test_split(all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
            else:
                trainX, trainY = all_images, all_labels
                testX, testY = all_images, all_labels 

            model = Sequential([
                tf.keras.Input(shape=(100, 100, 3)),
                Conv2D(32, (3,3), activation='relu'), MaxPooling2D(2,2), Dropout(0.2),
                Conv2D(64, (3,3), activation='relu'), MaxPooling2D(2,2), Dropout(0.2),
                Flatten(),
                Dense(128, activation='relu'), Dropout(0.5),
                Dense(num_classes if num_classes > 1 else 2, activation='softmax') 
            ])
            
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=1)
            
            model_dir = f'{base_path}/model'
            os.makedirs(model_dir, exist_ok=True)
            model.save(f'{model_dir}/face_cnn_model.keras')
            with open(f'{model_dir}/label_map.json', 'w') as f:
                json.dump(label_map, f)
            print("💾 บันทึกโมเดลเรียบร้อยแล้ว")
        else:
            print("❌ ไม่พบรูปภาพ")
            
        loss, acc = model.evaluate(testX, testY, verbose=0)
        print(f" Accuracy: {acc:.2f}")
        
        target_url = url_for('index', msg='registered')
        
        # ส่ง example_image กลับไปด้วย
        return jsonify({
            'status': 'success', 
            'redirect_url': target_url, 
            'example_image': example_image_b64
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'status': 'error', 'message': f"เกิดข้อผิดพลาด: {str(e)}"}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            model_path = f'{base_path}/model/face_cnn_model.keras'
            if not os.path.exists(model_path):
                return jsonify({'status': 'error', 'message': "ระบบยังไม่พร้อม (ยังไม่มีโมเดล)"})
                
            model = tf.keras.models.load_model(model_path)
            with open(f'{base_path}/model/label_map.json', 'r') as f:
                label_map = json.load(f)
            inv_label_map = {v: k for k, v in label_map.items()}
            
            phone = request.form['phone']
            file = request.files['image']
            
            users_df = pd.read_csv(csv_file, dtype={'phone': str})
            user_row = users_df[users_df['phone'] == phone]

            if user_row.empty:
                return jsonify({'status': 'error', 'message': f"ไม่พบเบอร์โทร {phone} ในระบบ"})

            expected_folder = os.path.basename(user_row.iloc[0]['folder'])
            
            img_cv = process_upload_to_cv2(file)
            cropped_face = detect_and_crop_face(img_cv)

            if cropped_face is None:
                return jsonify({'status': 'error', 'message': "ไม่พบใบหน้าในกล้อง กรุณาถ่ายใหม่ให้ชัดเจน"})

            img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            pred = model.predict(img_batch)
            pred_idx = np.argmax(pred)
            
            if pred_idx in inv_label_map:
                pred_label = inv_label_map[pred_idx]
            else:
                pred_label = "Unknown"

            if pred_label == expected_folder:
                session['phone'] = str(user_row.iloc[0]['phone'])
                session['name'] = user_row.iloc[0]['name']
                session['surname'] = user_row.iloc[0]['surname']
                
                has_voted = int(user_row.iloc[0]['has_voted'])
                target_url = url_for('results') if has_voted == 1 else url_for('vote_page')
                
                return jsonify({'status': 'success', 'redirect_url': target_url})
            else:
                return jsonify({'status': 'error', 'message': "ใบหน้าไม่ตรงกับข้อมูลในระบบ (Face Mismatch)"})

        except Exception as e:
            return jsonify({'status': 'error', 'message': f"Error: {str(e)}"})

    return render_template('index.html')

@app.route('/vote')
def vote_page():
    if 'phone' not in session: return redirect(url_for('index'))
    vote_df = pd.read_csv(vote_file)
    candidates = vote_df.to_dict('records')
    return render_template('vote.html', name=session.get('name'), surname=session.get('surname'), candidates=candidates)

@app.route('/submit_vote', methods=['POST'])
def submit_vote():
    if 'phone' not in session: return redirect(url_for('index'))
    phone = session['phone']
    candidate_id = int(request.form['candidate_id'])
    users_df = pd.read_csv(csv_file, dtype={'phone': str})
    idx = users_df.index[users_df['phone'] == phone].tolist()
    if not idx: return "User error", 400
    if users_df.at[idx[0], 'has_voted'] == 1: return "ใช้สิทธิ์ไปแล้ว!"
    users_df.at[idx[0], 'has_voted'] = 1
    users_df.to_csv(csv_file, index=False)
    vote_df = pd.read_csv(vote_file)
    c_idx = vote_df.index[vote_df['candidate_id'] == candidate_id].tolist()
    if c_idx:
        vote_df.at[c_idx[0], 'vote_count'] += 1
        vote_df.to_csv(vote_file, index=False)
    return redirect(url_for('results'))

@app.route('/results')
def results():
    vote_df = pd.read_csv(vote_file)
    vote_df = vote_df.sort_values(by='vote_count', ascending=False)
    rows = vote_df.to_dict('records')
    return render_template('result.html', rows=rows)

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        pwd = request.form['password']
        if pwd == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return "<h1>รหัสผ่านผิด! <a href='/admin'>ลองใหม่</a></h1>"
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    df = pd.read_csv(vote_file)
    candidates = df.to_dict('records')
    return render_template('admin_dashboard.html', candidates=candidates)

@app.route('/admin/add_candidate', methods=['POST'])
def add_candidate():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    name = request.form['candidate_name']
    df = pd.read_csv(vote_file)
    new_id = 1
    if not df.empty: new_id = df['candidate_id'].max() + 1
    new_row = pd.DataFrame([[new_id, name, 0]], columns=['candidate_id', 'candidate_name', 'vote_count'])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(vote_file, index=False)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_candidate', methods=['POST'])
def delete_candidate():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    c_id = int(request.form['id'])
    df = pd.read_csv(vote_file)
    df = df[df['candidate_id'] != c_id]
    df.to_csv(vote_file, index=False)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reset_votes', methods=['POST'])
def reset_votes():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    df_votes = pd.read_csv(vote_file)
    df_votes['vote_count'] = 0
    df_votes.to_csv(vote_file, index=False)
    df_users = pd.read_csv(csv_file, dtype={'phone': str})
    df_users['has_voted'] = 0
    df_users.to_csv(csv_file, index=False)
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/static/uploads/UserImages/<user_folder>/<image_name>')
def uploaded_file(user_folder, image_name):
    return send_from_directory(os.path.join(user_img_dir, user_folder), image_name)

NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(5000)
print(f"👉 Public URL: {public_url}")

if __name__ == '__main__':
    app.run(port=5000)