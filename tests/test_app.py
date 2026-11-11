import math
import unittest

import numpy as np

from app import FEATURES, load_model


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_pipeline_keeps_training_preprocessing(self):
        self.assertEqual(
            [step.__class__.__name__ for _, step in self.model.steps],
            ["MinMaxScaler", "LinearRegression"],
        )
        self.assertEqual(self.model.n_features_in_, len(FEATURES))

    def test_default_inputs_produce_expected_finite_prediction(self):
        sample = np.array(
            [[feature["default"] for feature in FEATURES]], dtype=float
        )

        prediction = float(self.model.predict(sample)[0])

        self.assertTrue(math.isfinite(prediction))
        self.assertAlmostEqual(prediction, 4.4679873, places=6)

    def test_model_is_cached(self):
        self.assertIs(load_model(), self.model)


if __name__ == "__main__":
    unittest.main()
