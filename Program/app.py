from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import sys
import webbrowser
from threading import Timer

app = Flask(__name__)

# fungsi pyinstaller
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# upload folder
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# load model
model_path = resource_path("best_model.keras")
model = tf.keras.models.load_model(model_path)


# nama kelas (kategori)
class_names = [
    "komersial",
    "pemukiman",
    "ruang_terbuka_hijau"
]


# halaman utama
@app.route("/")
def home():
    return render_template("home.html")


# halaman pembuat
@app.route("/pembuat")
def pembuat():
    return render_template("pembuat.html")


# halaman pengujian
@app.route("/pengujian", methods=["GET", "POST"])
def pengujian():

    hasil = None
    confidence = 0
    image_path = None
    risiko = ""
    karakteristik = []
    color = "#ffffff"

    if request.method == "POST":

        file = request.files["file"]

        if file and file.filename != "":

            filename = file.filename

            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(save_path)

            image_path = "/static/uploads/" + filename

            img = image.load_img(
                save_path,
                target_size=(224, 224)
            )

            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            probs = prediction[0]

            confidence = float(np.max(probs))

            hasil = class_names[
                np.argmax(probs)
            ]

            # warna kategori + risiko kebakaran + karakteristik wilayah
            if hasil == "pemukiman":
                color = "#A16207"
                risiko = "TINGGI 🔥"
                karakteristik = [
                    "Kepadatan bangunan tinggi",
                    "Vegetasi relatif rendah",
                    "Potensi risiko kebakaran tinggi"
                ]

            elif hasil == "komersial":
                color = "#F59E0B"
                risiko = "SEDANG ⚠️"
                karakteristik = [
                    "Aktivitas kawasan tinggi",
                    "Kepadatan bangunan sedang",
                    "Potensi risiko kebakaran sedang"
                ]

            else:
                color = "#22C55E"
                risiko = "RENDAH 🌿"
                karakteristik = [
                    "Dominasi vegetasi tinggi",
                    "Kepadatan bangunan rendah",
                    "Potensi risiko kebakaran rendah"
                ]

    return render_template(
        "pengujian.html",
        hasil=hasil,
        confidence=confidence,
        image_path=image_path,
        color=color,
        risiko=risiko,
        karakteristik=karakteristik
    )


# browser otomatis
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


# run app
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )