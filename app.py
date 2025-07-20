import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk

# Load models
age_model = load_model('./1.1_age_input_output/output/age_model_pretrained.h5')
gender_model = load_model('./1.2_gender_input_output/output/gender_model_pretrained.h5')
emotion_model = load_model('./1.3_emotion_input_output/output/emotion_model_pretrained.h5')

# Labels
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges = ['positive', 'negative', 'neutral']

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('./1.4_test_input/cv2_cascade_classifier/haarcascade_frontalface_default.xml')

# ------------------ Detection Functions ------------------

def predict_face_attributes(face_gray):
    # Emotion (48x48)
    emotion_img = cv2.resize(face_gray, (48, 48))
    emotion_input = np.expand_dims(np.expand_dims(emotion_img, -1), 0)
    emotion = emotion_ranges[np.argmax(emotion_model.predict(emotion_input))]

    # Gender (100x100)
    gender_img = cv2.resize(face_gray, (100, 100))
    gender_input = np.expand_dims(np.expand_dims(gender_img, -1), 0)
    gender = gender_ranges[np.argmax(gender_model.predict(gender_input))]

    # Age (200x200)
    age_img = cv2.resize(face_gray, (200, 200))
    age_input = np.expand_dims(np.expand_dims(age_img, -1), 0)
    age = age_ranges[np.argmax(age_model.predict(age_input))]

    return gender, age, emotion

def detect_from_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return

    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        messagebox.showinfo("Result", "No faces detected.")
        return

    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_gray = gray[y:y + h, x:x + w]
        gender, age, emotion = predict_face_attributes(face_gray)

        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (203, 12, 255), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result in a new window
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    result_window = tk.Toplevel(root)
    result_window.title("Detection Result")
    lbl = tk.Label(result_window, image=img_tk)
    lbl.image = img_tk
    lbl.pack()

def detect_from_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not detected!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces, start=1):
            face_gray = gray[y:y + h, x:x + w]
            gender, age, emotion = predict_face_attributes(face_gray)

            label = f"{gender}, {age}, {emotion}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (203, 12, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Stream Detection (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ GUI ------------------

root = tk.Tk()
root.title("Age, Gender & Emotion Detection")
root.geometry("400x200")

tk.Label(root, text="Choose Detection Mode", font=("Arial", 16)).pack(pady=20)

tk.Button(root, text="Image Detection", command=detect_from_image, width=25, height=2).pack(pady=10)
tk.Button(root, text="Stream Detection", command=detect_from_stream, width=25, height=2).pack(pady=10)

root.mainloop()
