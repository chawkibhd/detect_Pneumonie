import tkinter as tk
from tkinter import filedialog,ttk
from PIL import Image, ImageTk
import numpy as np
import joblib
import tensorflow as tf 
import os
from tensorflow.keras.preprocessing import image
from ImageFeatureExtractor import ImageFeatureExtractor

class PneumoniaDetectorApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Détecteur de Pneumonie")
        self.pack()

        self.extractor = ImageFeatureExtractor(
            base_dir=".", 
            csv_file_path="features.csv",
            target_size=(128, 128)
        )

        #self.model = joblib.load("/Users/chawkibhd/Desktop/LR_grabor/logistic_regression_model.pkl")
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'logistic_regression_model.pkl')
        self.model = joblib.load(model_path)

        model_path2 = os.path.join(os.path.dirname(__file__), 'models', 'logreg-v3.pkl')
        self.model_two = joblib.load(model_path2)

        self.selected_image_path = None
        self.file_path = None
        
        self.create_widgets()

    def create_widgets(self):

        self.image_label = tk.Label(self, text="Aucune image sélectionnée", bg="white")
        self.image_label.pack(pady=10)

        self.browse_button = tk.Button(self, text="Choisir une image", command=self.browse_image)
        self.browse_button.pack(pady=5)

        self.selected_var = tk.StringVar()
        self.selected_var.set("CNN") 

        options = ["CNN","CNN", "LR_2features", "LR_5features"]

        self.dropdown = ttk.OptionMenu(self, self.selected_var, *options)
        self.dropdown.pack(pady=20)

        self.predict_button = tk.Button(self, text="Prédire", command=self._predict)
        self.predict_button.pack(pady=5)

        self.result_label = tk.Label(self, text="", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        self.file_path = file_path
        
        if file_path:
            self.selected_image_path = file_path

            pil_image = Image.open(file_path)
            pil_image = pil_image.resize((200, 200))
            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=tk_image, text="")
            self.image_label.image = tk_image
        else:
            self.selected_image_path = None
            self.image_label.config(text="Aucune image sélectionnée", image="")

    def _predict(self):
        if not self.selected_image_path:
            self.result_label.config(text="Veuillez sélectionner une image d'abord.", fg="red")
            return

        image = self.extractor.load_and_preprocess_image(self.selected_image_path)

        gabor_feats = self.extractor.compute_gabor(image)
        dct_feats   = self.extractor.compute_dct(image, num_coeffs=20)
        ft_feats    = self.extractor.compute_ft(image,  num_coeffs=20)
        phog_feats  = self.extractor.compute_phog(image, num_coeffs=20)

        gabor_mean = np.mean(gabor_feats)
        dct_mean   = np.mean(dct_feats)
        ft_mean    = np.mean(ft_feats)
        phog_mean  = np.mean(phog_feats)

        feature_vector = np.array([gabor_mean, dct_mean, ft_mean, phog_mean]).reshape(1, -1)

        prediction = self.get_prediction(feature_vector)
        self.update_result(prediction)

    def predict_two(self): 
        if not self.selected_image_path: 
            self.result_label.config(text="Veuillez sélectionner une image d'abord.", fg="red") 
            return 
 
        image = self.extractor.load_and_preprocess_image(self.selected_image_path) 
 
        gabor_feats = self.extractor.compute_gabor(image) 
        dct_feats   = self.extractor.compute_dct(image, num_coeffs=20) 
        ft_feats    = self.extractor.compute_ft(image,  num_coeffs=20) 
        phog_feats  = self.extractor.compute_phog(image, num_coeffs=20) 
 
        all_feats = np.concatenate([gabor_feats, dct_feats, ft_feats, phog_feats]) 
                         
        feature_vector = np.mean(all_feats).reshape(1, -1) 
 
        prediction = self.get_prediction(feature_vector) 
        self.update_result(prediction)

    def detect_using_cnn(self, image_path: str) -> bool: 
        # 1) Load the model 
        #model = tf.keras.models.load_model("/Users/chawkibhd/Desktop/LR_grabor/my_model-v4-bestOne.h5")
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'my_model-v4-bestOne.h5')
        model = tf.keras.models.load_model(model_path)
     
        # 2) Define image path and target size 
        img_height, img_width = 128, 128 
     
        # 3) Preprocess the image 
        test_img = image.load_img(image_path, target_size=(img_height, img_width)) 
        img_array = image.img_to_array(test_img) 
        img_array = np.expand_dims(img_array, axis=0) 
     
     
        # 4) scale pixel values to [0,1] 
        img_array = img_array / 255.0 
     
        predictions = model.predict(img_array) 
        print("Raw output:", predictions) 
     
        # Binary classification w/ sigmoid output 
        # True -> Class (1) (PNEUMONIA) 
        # False -> Class 0 (NORMAL) 
        return True if predictions[0][0] > 0.5 else False

    def get_prediction(self, feature_vector):
        selected_option = self.selected_var.get()
        if selected_option == "CNN":
            return self.detect_using_cnn(self.file_path)
        elif selected_option == "LR_2features":
            return self.model_two.predict(self.predict_two())[0]
        elif selected_option == "LR_5features":
            return self.model.predict(self._predict())[0]

    def update_result(self, prediction):
        if prediction == 0:
            self.result_label.config(text="Résultat : CAS NORMAL", fg="green")
        else:
            self.result_label.config(text="Résultat : PNEUMONIE", fg="red")


def main():
    root = tk.Tk()
    root.geometry("600x400") 
    window_width = 600
    window_height = 400

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)

    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    app = PneumoniaDetectorApp(master=root)
    app.mainloop()

if __name__ == "__main__":
    main()