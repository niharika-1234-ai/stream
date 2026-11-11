import logging
import math
from pathlib import Path

import pandas as pd
import skops.io as sio
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "trained_model.skops"
TARGET_NAME = "MedHouseVal"

# Feature order and observed input ranges come from the existing California
# Housing model API and its fitted MinMaxScaler. Keep this order when predicting.
FEATURES = (
    {
        "name": "MedInc",
        "label": "Median income",
        "help": "Median household income, in tens of thousands of dollars.",
        "minimum": 0.4999,
        "maximum": 15.0001,
        "default": 8.3,
        "step": 0.1,
    },
    {
        "name": "HouseAge",
        "label": "House age",
        "help": "Median age of houses in the block group, in years.",
        "minimum": 1.0,
        "maximum": 52.0,
        "default": 25.0,
        "step": 1.0,
    },
    {
        "name": "AveRooms",
        "label": "Average rooms",
        "help": "Average number of rooms per household.",
        "minimum": 0.8888888888888888,
        "maximum": 141.9090909090909,
        "default": 5.0,
        "step": 0.1,
    },
    {
        "name": "AveBedrms",
        "label": "Average bedrooms",
        "help": "Average number of bedrooms per household.",
        "minimum": 0.3333333333333333,
        "maximum": 25.636363636363637,
        "default": 1.0,
        "step": 0.1,
    },
    {
        "name": "Population",
        "label": "Population",
        "help": "Total population of the block group.",
        "minimum": 3.0,
        "maximum": 35682.0,
        "default": 300.0,
        "step": 1.0,
    },
    {
        "name": "AveOccup",
        "label": "Average occupancy",
        "help": "Average number of household members.",
        "minimum": 0.6923076923076923,
        "maximum": 1243.3333333333333,
        "default": 3.0,
        "step": 0.1,
    },
    {
        "name": "Latitude",
        "label": "Latitude",
        "help": "Block group latitude.",
        "minimum": 32.55,
        "maximum": 41.95,
        "default": 37.0,
        "step": 0.01,
    },
    {
        "name": "Longitude",
        "label": "Longitude",
        "help": "Block group longitude.",
        "minimum": -124.35,
        "maximum": -114.31,
        "default": -122.0,
        "step": 0.01,
    },
)


@st.cache_resource
def load_model():
    """Load the bundled, pre-trained model once for the Streamlit process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved model was not found at {MODEL_PATH}. "
            "Ensure model/trained_model.skops is present."
        )

    try:
        return sio.load(MODEL_PATH, trusted=[])
    except Exception as exc:
        logging.getLogger(__name__).exception("Could not load the saved model.")
        raise RuntimeError(
            "The saved model could not be loaded. Check the model file and "
            "the installed scikit-learn/skops versions."
        ) from exc


def main():
    st.set_page_config(
        page_title="California Housing Predictor",
        page_icon="🏡",
        layout="centered",
    )
    st.title("California Housing Price Predictor")
    st.write(
        "Enter block-group details to estimate its median house value. "
        "The saved model includes the same preprocessing pipeline used "
        "during training."
    )
    st.caption(
        "Values are limited to the ranges observed by the model during "
        "training. The estimate is in hundreds of thousands of US dollars."
    )

    try:
        model = load_model()
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Block group details")
        columns = st.columns(2)
        input_values = {}

        for index, feature in enumerate(FEATURES):
            with columns[index % 2]:
                input_values[feature["name"]] = st.number_input(
                    feature["label"],
                    min_value=feature["minimum"],
                    max_value=feature["maximum"],
                    value=feature["default"],
                    step=feature["step"],
                    help=feature["help"],
                    key=feature["name"],
                )

        submitted = st.form_submit_button(
            "Predict median house value", type="primary", use_container_width=True
        )

    if not submitted:
        return

    for feature in FEATURES:
        value = input_values[feature["name"]]
        if not math.isfinite(value):
            st.error(f"{feature['label']} must be a finite number.")
            return
        if not feature["minimum"] <= value <= feature["maximum"]:
            st.error(
                f"{feature['label']} must be between "
                f"{feature['minimum']:g} and {feature['maximum']:g}."
            )
            return

    input_frame = pd.DataFrame(
        [input_values], columns=[feature["name"] for feature in FEATURES]
    )

    try:
        prediction = float(model.predict(input_frame.to_numpy(dtype=float))[0])
        if not math.isfinite(prediction):
            raise ValueError("The model returned a non-finite prediction.")
    except Exception:
        logging.getLogger(__name__).exception("Model prediction failed.")
        st.error("Prediction failed. Check the entered values and try again.")
        return

    st.success("Prediction completed.")
    st.metric(
        label=f"Estimated median house value ({TARGET_NAME})",
        value=f"${prediction * 100_000:,.0f}",
        help=f"Raw model output: {prediction:.4f} (hundreds of thousands of dollars).",
    )


if __name__ == "__main__":
    main()
