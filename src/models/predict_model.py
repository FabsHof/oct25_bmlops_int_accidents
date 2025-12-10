from src.utils import logging
import argparse

def predict_model(model_path: str, input_data: dict) -> dict:
    logging.info('Predicting with model at %s on input data: %s.', model_path, input_data)
    pass

def main(model_path: str, input_data: dict) -> dict:
    predict_model(model_path, input_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use the trained model to make predictions on the severity of a road accident.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--input_data', type=str, required=True, help='Input data for prediction in JSON format.')
    args = parser.parse_args()
    main(args.model_path, args.input_data)