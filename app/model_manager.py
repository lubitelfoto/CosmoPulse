from catboost import CatBoostClassifier


class ModelManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = CatBoostClassifier()
        self.model.load_model('model.cbm')
        print("Model loaded")

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.model