import mlrun
import mlrun.serving
import numpy as np
import cloudpickle 

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        model_file, extra_data = self.get_model(".pkl")
        self.model = cloudpickle.load(open(model_file, "rb"))
        print("Model loaded successfully.")
        self.ready = True 

    def predict(self, body: dict) -> list:
        if not self.ready:
            raise RuntimeError("Model is not loaded or ready.")

        try:
            input_features = np.asarray(body["inputs"])
            predictions = self.model.predict(input_features)

            return predictions.tolist()

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {e}")
