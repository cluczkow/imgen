# imgen.py

"""
This is the main module of the imgen application. It contains core functionality related to image generation.
"""

import logging
import argparse
import json
import os
import random
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib  # For saving and loading the trained model
from PIL import Image, ImageDraw

# Import procedural routines from generator.py
from generator import (
    generate_gradient,
    generate_noise,
    generate_circles,
    generate_stripes,
    generate_radial_gradient,
    generate_perlin_noise,
    generate_checkerboard,
    generate_fractal,
    generate_wave,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_feedback(feedback, feedback_file="feedback.json"):
    """Save user feedback to a JSON file."""
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as file:
            data = json.load(file)
    else:
        data = []

    data.append(feedback)

    with open(feedback_file, "w") as file:
        json.dump(data, file, indent=4)

def train_model(feedback_file="feedback.json", model_file="model.pkl"):
    """Train a model based on user feedback."""
    if not os.path.exists(feedback_file):
        print(f"No feedback file found at {feedback_file}. Please provide feedback first.")
        return

    with open(feedback_file, "r") as file:
        data = json.load(file)

    # Prepare training data
    X = []  # Features
    y = []  # Labels
    for entry in data:
        features = [
            len(entry["parameters"]["routines"]),  # Number of routines
            sum(entry["parameters"]["opacity"]) / len(entry["parameters"]["opacity"]),  # Avg opacity
            entry["parameters"]["num_layers"],  # Number of layers
        ]
        X.append(features)
        y.append(1 if entry["liked"] else 0)

    # Train the model
    model = RandomForestClassifier()
    model.fit(np.array(X), np.array(y))

    # Save the trained model
    joblib.dump(model, model_file)
    print(f"Model trained and saved to {model_file}.")

def predict_likelihood(model, parameters):
    """Predict the likelihood of the user liking an image."""
    features = [
        len(parameters["routines"]),
        sum(parameters["opacity"]) / len(parameters["opacity"]),
        parameters["num_layers"],
    ]
    return model.predict_proba([features])[0][1]  # Probability of "like"

def generate_image_with_feedback(width, height, model, routines):
    """Generate an image guided by user feedback."""
    best_score = -1
    best_parameters = None

    # Generate multiple candidates
    for _ in range(10):  # Generate 10 candidates
        parameters = {
            "routines": random.sample(list(routines.keys()), random.randint(3, 6)),
            "opacity": [random.randint(50, 200) for _ in range(random.randint(3, 6))],
            "num_layers": random.randint(3, 6),
        }
        score = predict_likelihood(model, parameters)
        if score > best_score:
            best_score = score
            best_parameters = parameters

    # Generate the image using the best parameters
    output_path, _ = generate_image_with_parameters(width, height, best_parameters, routines)
    return output_path, best_parameters, best_score

def generate_image_with_parameters(width, height, parameters, routines):
    """Generate an image using specific parameters."""
    output_dir = "out"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = f"{timestamp}.png"
    output_path = os.path.join(output_dir, filename)

    base_image = Image.new("RGBA", (width, height), (0, 0, 0, 255))

    for routine_name, opacity in zip(parameters["routines"], parameters["opacity"]):
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        routines[routine_name](draw, width, height, opacity)
        base_image = Image.alpha_composite(base_image, layer)

    final_image = base_image.convert("RGB")
    final_image.save(output_path)

    return output_path, parameters

def main():
    logging.info("Starting the imgen CLI...")

    parser = argparse.ArgumentParser(description="Image generation CLI with feedback learning.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: Generate images
    generate_parser = subparsers.add_parser("generate", help="Generate images")
    generate_parser.add_argument("--width", type=int, default=1920, help="Width of the image (default: 1920)")
    generate_parser.add_argument("--height", type=int, default=1080, help="Height of the image (default: 1080)")
    generate_parser.add_argument("-n", "--count", type=int, default=1, help="Number of images to generate (default: 1)")

    # Command: Train model
    train_parser = subparsers.add_parser("train", help="Train the model based on feedback")
    train_parser.add_argument("--feedback", type=str, default="feedback.json", help="Path to the feedback file (default: feedback.json)")
    train_parser.add_argument("--model", type=str, default="model.pkl", help="Path to save the trained model (default: model.pkl)")

    # Command: Generate guided images
    guided_parser = subparsers.add_parser("generate-guided", help="Generate images guided by user feedback")
    guided_parser.add_argument("--width", type=int, default=1920, help="Width of the image (default: 1920)")
    guided_parser.add_argument("--height", type=int, default=1080, help="Height of the image (default: 1080)")
    guided_parser.add_argument("-n", "--count", type=int, default=1, help="Number of images to generate (default: 1)")
    guided_parser.add_argument("--model", type=str, default="model.pkl", help="Path to the trained model (default: model.pkl)")

    args = parser.parse_args()

    if args.command == "generate":
        for _ in range(args.count):
            output_path, parameters = generate_image(args.width, args.height)
            liked = input(f"Did you like the image {output_path}? (yes/no): ").strip().lower() == "yes"
            feedback = {
                "image_path": output_path,
                "liked": liked,
                "parameters": parameters,
            }
            save_feedback(feedback)
            logging.info(f"Image generated: {output_path}")
    elif args.command == "train":
        train_model(feedback_file=args.feedback, model_file=args.model)
    elif args.command == "generate-guided":
        if not os.path.exists(args.model):
            print(f"Trained model not found at {args.model}. Please train the model first.")
            return

        model = joblib.load(args.model)
        routines = {
            "gradient": generate_gradient,
            "noise": generate_noise,
            "circles": generate_circles,
            "stripes": generate_stripes,
            "radial_gradient": generate_radial_gradient,
            "perlin_noise": generate_perlin_noise,
            "checkerboard": generate_checkerboard,
            "fractal": generate_fractal,
            "wave": generate_wave,
        }

        for _ in range(args.count):
            output_path, parameters, score = generate_image_with_feedback(args.width, args.height, model, routines)
            print(f"Generated guided image: {output_path} (Score: {score:.2f})")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()