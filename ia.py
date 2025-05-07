import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class StressDetector:
    def __init__(self):
        # On prépare le scaler et le modèle
        self.scaler = StandardScaler()
        self.model = LogisticRegression()

    def prepare_data(self, df: pd.DataFrame):
        """
        df doit contenir trois colonnes :
          - 'pulse' : fréquence cardiaque en BPM
          - 'humidity' : humidité de la peau normalisée (0 à 1)
          - 'stress' : label binaire (0 = pas stressé, 1 = stressé)
        """
        X = df[['pulse', 'humidity']].values
        y = df['stress'].values
        # On met à l’échelle les deux capteurs pour un apprentissage plus stable
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîne le modèle sur les données préparées."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Affiche des métriques de performance."""
        y_pred = self.model.predict(X_test)
        print("Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5):
        """Optionnel : score moyen en validation croisée."""
        scores = cross_val_score(self.model, X, y, cv=cv)
        print(f"Accuracy moyen ({cv}-fold) : {scores.mean():.2f} ± {scores.std():.2f}")

    def predict(self, pulse: float, humidity: float) -> bool:
        """
        Prédit si la personne est stressée (True) ou non (False)
        à partir de deux valeurs brutes.
        """
        x = np.array([[pulse, humidity]])
        x_scaled = self.scaler.transform(x)
        return bool(self.model.predict(x_scaled)[0])


if __name__ == "__main__":
    # --- 1) Chargement d'un jeu de données d'exemple ---
    # Ici, on simule un petit jeu de données ; en pratique,
    # remplacez par vos mesures réelles et leurs étiquettes.
    data = {
        'pulse':     [60, 65, 80, 90, 100, 55, 75, 85, 95, 105, 70, 110, 115, 50, 68],
        'humidity':  [0.20,0.22,0.35,0.40,0.45,0.18,0.30,0.37,0.42,0.50,0.28,0.55,0.60,0.15,0.25],
        'stress':    [  0,   0,   0,   1,   1,   0,   0,   1,   1,   1,   0,   1,   1,   0,   0]
    }
    df = pd.DataFrame(data)

    # --- 2) Préparation, entraînement et évaluation ---
    detector = StressDetector()
    X_train, X_test, y_train, y_test = detector.prepare_data(df)
    detector.train(X_train, y_train)
    detector.evaluate(X_test, y_test)

    # (Optionnel) validation croisée sur tout le jeu de données
    X_all = detector.scaler.transform(df[['pulse','humidity']].values)
    detector.cross_validate(X_all, df['stress'].values, cv=4)

    # --- 3) Prédiction sur de nouvelles mesures ---
    test_pulse = 88   # BPM mesuré
    test_humidity = 0.38  # humidité mesurée
    is_stressed = detector.predict(test_pulse, test_humidity)
    print(f"\nPour pulse={test_pulse} BPM et humidité={test_humidity:.2f}, "
          f"stress détecté ? {is_stressed}")
