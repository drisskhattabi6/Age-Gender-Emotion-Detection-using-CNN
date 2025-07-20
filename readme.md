# Age, Gender & Emotion Detection App

A Tkinter-based GUI application that detects faces and predicts **age**, **gender**, and **emotion** on static images or live webcam stream using pretrained CNN models.

---

## Project Structure

```

├── Notebooks/    # Jupyter notebooks for training each model
│   ├── 2.1\_train\_age\_model.ipynb
│   ├── 2.2\_train\_gender\_model.ipynb
│   ├── 2.3\_train\_emotion\_model.ipynb
│   └── 3.1\_Pred\_Final.ipynb
├── app.py       # Main Tkinter application entry point.
├── imgs/       # Example images to test the “Image Detection” mode.
└── models/     # Pretrained Keras models and OpenCV Haar cascade for face detection.
    ├── age\_model\_pretrained.h5
    ├── emotion\_model\_pretrained.h5
    ├── gender\_model\_pretrained.h5
    └── haarcascade\_frontalface\_default.xml
````


---

## Approach

1. **Face Detection**  
   - OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).

2. **Model Training**  
   - **2.1_train_age_model.ipynb**: CNN trained on Facial-age & UTKFace  
   - **2.2_train_gender_model.ipynb**: CNN trained on UTKFace  
   - **2.3_train_emotion_model.ipynb**: CNN trained on CT+  

3. **Final Prediction**  
   - **3.1_Pred_Final.ipynb**: notebook demonstrating inference on static images.

4. **GUI App**  
   - **app.py**: lets the user choose between image or webcam-based detection.

---

## Datasets

> **Note:** Datasets are **not included** in this repository due to size constraints.  
You will need to download and prepare them yourself:

- **Facial-age & UTKFace** for **age detection**  
- **UTKFace** for **gender detection**  
- **CT+** for **emotion detection**  

Once downloaded, preprocess and train the notebooks in the **Notebooks/** folder. The pretrained `.h5` files in **models/** reflect those trainings.

---

## Labels & Performance

- **Age Ranges:**  
  `['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']` — **82.5%** accuracy

- **Gender Classes:**  
  `['male', 'female']` — **89.5%** accuracy

- **Emotion Classes:**  
  `['positive', 'negative', 'neutral']` — **97.2%** accuracy

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/age-gender-emotion-app.git
   cd age-gender-emotion-app
````

2. **Install dependencies**

   ```bash
   pip install opencv-python-headless tensorflow pillow numpy
   ```

3. **Verify files**

   * Models in `models/`
   * Cascade XML in `models/haarcascade_frontalface_default.xml`

---

## Usage

```bash
python app.py
```

1. **Image Detection**

   * Click **Image Detection**
   * Browse and select an image from `imgs/` or elsewhere
   * A window will pop up showing detected faces with labels.

2. **Stream Detection**

   * Click **Stream Detection**
   * Your webcam opens in a window; press **q** to quit.

---

## Screenshots

*Add your own screenshots here:*

<img src="imgs/family.jpg" alt="Image Detection Example" width="300"/>  
*Static image detection*

> *Replace with actual annotated screenshots.*

"# Age-Gender-Emotion-Detection-using-CNN" 
