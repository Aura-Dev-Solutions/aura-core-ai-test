import joblib
import os

# Inicializamos modelo
_loaded_model = None

def _load_model():
    global _loaded_model
    if _loaded_model is None:
        path = "app/models/classifier_model.pkl"
        if not os.path.isfile(path):
            raise FileNotFoundError("No se encuentra el modelo de clasificación en: " + path)
        _loaded_model = joblib.load(path)
    return _loaded_model

def classify(text: str):
    """
    Clasifica el texto usando un modelo preentrenado.
    :param text: Texto a clasificar
    :return: clase del documento
    """
    model = _load_model()
    return model.predict([text])[0]