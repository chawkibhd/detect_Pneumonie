import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import joblib
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from core.ImageFeatureExtractor import ImageFeatureExtractor

class PneumoniaDetectorApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Détecteur de Pneumonie")
        self.pack()

        # Initialisation de l'extracteur de caractéristiques
        self.extractor = ImageFeatureExtractor(
            base_dir=".", 
            csv_file_path="features.csv",
            target_size=(128, 128)
        )

        # Chargement des modèles de régression logistique
        models_dir = os.path.join(os.getcwd(), 'models')  # Utilisation de os.getcwd() pour la portabilité

        # Chargement des modèles avec vérification de l'existence
        self.model_lr_4features = self.load_model(models_dir, 'logistic_regression_multi_features_model.pkl')
        self.model_lr_1feature = self.load_model(models_dir, 'logistic_regression_single_feature_model.pkl')
        self.cnn_model_path = os.path.join(models_dir, 'cnn_model.h5')

        self.selected_image_path = None
        self.file_path = None
        
        self.create_widgets()

    def load_model(self, models_dir, model_filename):
        model_path = os.path.join(models_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable à l'emplacement : {model_path}")
        print(f"Chargement du modèle : {model_path}")
        return joblib.load(model_path)

    def create_widgets(self):
        # Étiquette pour afficher l'image sélectionnée
        self.image_label = tk.Label(self, text="Aucune image sélectionnée", bg="white")
        self.image_label.pack(pady=10)

        # Bouton pour parcourir et sélectionner une image
        self.browse_button = tk.Button(self, text="Choisir une image", command=self.browse_image)
        self.browse_button.pack(pady=5)

        # Variable et menu déroulant pour sélectionner le modèle
        self.selected_var = tk.StringVar()
        self.selected_var.set("CNN") 

        options = ["CNN", "LR_1feature", "LR_4features"]

        self.dropdown = ttk.OptionMenu(self, self.selected_var, options[0], *options)
        self.dropdown.pack(pady=20)

        # Bouton pour lancer la prédiction
        self.predict_button = tk.Button(self, text="Prédire", command=self._predict)
        self.predict_button.pack(pady=5)

        # Étiquette pour afficher le résultat de la prédiction
        self.result_label = tk.Label(self, text="", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=10)

    def browse_image(self):
        # Ouvrir une boîte de dialogue pour sélectionner une image
        file_path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        self.file_path = file_path
        
        if file_path:
            self.selected_image_path = file_path

            # Afficher l'image sélectionnée dans l'interface
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

        try:
            selected_option = self.selected_var.get()
            print(f"Option sélectionnée : {selected_option}")

            if selected_option == "CNN":
                # Prédiction en utilisant le modèle CNN
                prediction = self.detect_using_cnn(self.file_path)
            elif selected_option == "LR_1feature":
                # Extraction de 1 caractéristique pour le modèle LR_1feature
                feature_vector = self.extract_features(num_features=1)
                prediction = self.get_prediction(feature_vector, model_type='LR_1feature')
            elif selected_option == "LR_4features":
                # Extraction de 4 caractéristiques pour le modèle LR_4features
                feature_vector = self.extract_features(num_features=4)
                prediction = self.get_prediction(feature_vector, model_type='LR_4features')
            else:
                self.result_label.config(text="Option non valide sélectionnée.", fg="red")
                return

            self.update_result(prediction)

        except Exception as e:
            self.result_label.config(text=f"Erreur : {str(e)}", fg="red")
            print(f"Erreur lors de la prédiction : {e}")

    def extract_features(self, num_features):
        image = self.extractor.load_and_preprocess_image(self.selected_image_path)
        print(f"Image chargée et prétraitée : {self.selected_image_path}")

        if num_features == 1:
            # Extraction de 1 caractéristique
            gabor_feats = self.extractor.compute_gabor(image)
            dct_feats = self.extractor.compute_dct(image, num_coeffs=20)
            ft_feats = self.extractor.compute_ft(image, num_coeffs=20)
            phog_feats = self.extractor.compute_phog(image, num_coeffs=20)

            all_feats = np.concatenate([gabor_feats, dct_feats, ft_feats, phog_feats])

            # Aggregate features into a single mean
            aggregated_feature = np.mean(all_feats)
            feature_vector = np.array([aggregated_feature]).reshape(1, -1)
            print(f"Feature vector (1 feature) : {feature_vector}")
            return feature_vector

        elif num_features == 4:
            # Extraction de 4 caractéristiques
            gabor_feats = self.extractor.compute_gabor(image)
            dct_feats = self.extractor.compute_dct(image, num_coeffs=20)
            ft_feats = self.extractor.compute_ft(image, num_coeffs=20)
            phog_feats = self.extractor.compute_phog(image, num_coeffs=20)
            gabor_mean = np.mean(gabor_feats)
            dct_mean = np.mean(dct_feats)
            ft_mean = np.mean(ft_feats)
            phog_mean = np.mean(phog_feats)
            feature_vector = np.array([gabor_mean, dct_mean, ft_mean, phog_mean]).reshape(1, -1)
            print(f"Feature vector (4 features) : {feature_vector}")
            return feature_vector

        else:
            raise ValueError("Nombre de caractéristiques non supporté.")

    def detect_using_cnn(self, image_path: str) -> int: 
        # Vérifiez si le modèle CNN existe
        if not os.path.exists(self.cnn_model_path):
            raise FileNotFoundError(f"Modèle CNN introuvable à l'emplacement : {self.cnn_model_path}")

        print(f"Chargement du modèle CNN : {self.cnn_model_path}")
        model = tf.keras.models.load_model(self.cnn_model_path)
     
        # Prétraitement de l'image
        img_height, img_width = 128, 128 
        test_img = image.load_img(image_path, target_size=(img_height, img_width)) 
        img_array = image.img_to_array(test_img) 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array / 255.0  # Normalisation des pixels
     
        predictions = model.predict(img_array) 
        print("Raw output CNN:", predictions) 
     
        # Classification binaire avec une sortie sigmoid
        # 1 -> Pneumonie
        # 0 -> Normal
        return 1 if predictions[0][0] > 0.5 else 0

    def get_prediction(self, feature_vector, model_type='LR_1feature'):
        """
        Effectue la prédiction en fonction du type de modèle sélectionné.
        Retourne 0 pour Normal et 1 pour Pneumonie.
        """
        print(f"Obtention de la prédiction pour le modèle : {model_type}")
        print(f"Feature vector shape : {feature_vector.shape}")
        print(f"Feature vector : {feature_vector}")

        if model_type == 'LR_1feature':
            if feature_vector.shape[1] != 1:
                raise ValueError(f"Modèle LR_1feature attend 1 caractéristique, mais {feature_vector.shape[1]} ont été fournies.")
            prediction = self.model_lr_1feature.predict(feature_vector)[0]
            print(f"Prédiction LR_1feature : {prediction}")
            return int(prediction)
        elif model_type == 'LR_4features':
            if feature_vector.shape[1] != 4:
                raise ValueError(f"Modèle LR_4features attend 4 caractéristiques, mais {feature_vector.shape[1]} ont été fournies.")
            prediction = self.model_lr_4features.predict(feature_vector)[0]
            print(f"Prédiction LR_4features : {prediction}")
            return int(prediction)
        else:
            raise ValueError("Type de modèle invalide.")

    def update_result(self, prediction):
        """
        Met à jour l'étiquette de résultat en fonction de la prédiction.
        """
        print(f"Résultat de la prédiction : {prediction}")
        if prediction == 0:
            self.result_label.config(text="Résultat : CAS NORMAL", fg="green")
        else:
            self.result_label.config(text="Résultat : PNEUMONIE", fg="red")