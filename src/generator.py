# filepath: c:\dev\imgen\src\generator.py
"""
This module contains the logic for generating images.
"""

import os
import math
import random
import time
from PIL import Image, ImageDraw
from perlin_noise import PerlinNoise

def generate_gradient(draw, width, height, opacity):
    """Generate a gradient layer."""
    for x in range(width):
        for y in range(height):
            red = int((x / width) * 255)
            green = int((y / height) * 255)
            blue = random.randint(0, 255)
            draw.point((x, y), fill=(red, green, blue, opacity))


def generate_noise(draw, width, height, opacity):
    """Generate a noise layer with reduced intensity."""
    for x in range(width):
        for y in range(height):
            red = random.randint(100, 150)
            green = random.randint(100, 150)
            blue = random.randint(100, 150)
            draw.point((x, y), fill=(red, green, blue, opacity))


def generate_circles(draw, width, height, opacity):
    """Generate random circles."""
    for _ in range(random.randint(5, 15)):
        radius = random.randint(10, 100)
        x = random.randint(0, width)
        y = random.randint(0, height)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            opacity,
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def generate_stripes(draw, width, height, opacity):
    """Generate random stripes."""
    stripe_width = random.randint(10, 50)
    for x in range(0, width, stripe_width * 2):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            opacity,
        )
        draw.rectangle((x, 0, x + stripe_width, height), fill=color)


def generate_radial_gradient(draw, width, height, opacity):
    """Generate a radial gradient layer."""
    center_x, center_y = width // 2, height // 2
    max_distance = ((center_x) ** 2 + (center_y) ** 2) ** 0.5
    for x in range(width):
        for y in range(height):
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            intensity = int((1 - (distance / max_distance)) * 255)
            red = intensity
            green = intensity
            blue = random.randint(0, 255)  # Add some randomness to the blue channel
            draw.point((x, y), fill=(red, green, blue, opacity))


def generate_perlin_noise(draw, width, height, opacity):
    """Generate a Perlin noise layer using the perlin-noise library."""
    noise = PerlinNoise(octaves=4, seed=random.randint(0, 1000))
    for x in range(width):
        for y in range(height):
            value = int((noise([x / width, y / height]) + 1) * 127.5)  # Normalize to 0–255
            draw.point((x, y), fill=(value, value, value, opacity))


def generate_checkerboard(draw, width, height, opacity):
    """Generate a checkerboard pattern layer."""
    square_size = random.randint(20, 50)
    for x in range(0, width, square_size):
        for y in range(0, height, square_size):
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                opacity if (x // square_size + y // square_size) % 2 == 0 else 0,
            )
            draw.rectangle((x, y, x + square_size, y + square_size), fill=color)


def generate_fractal_tree(draw, x, y, angle, depth, length, opacity):
    """Recursive function to draw a fractal tree."""
    if depth == 0:
        return
    x_end = x + int(length * math.cos(math.radians(angle)))
    y_end = y - int(length * math.sin(math.radians(angle)))
    color = (
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255),
        opacity,
    )
    draw.line((x, y, x_end, y_end), fill=color, width=2)
    generate_fractal_tree(draw, x_end, y_end, angle - random.randint(15, 30), depth - 1, length * 0.7, opacity)
    generate_fractal_tree(draw, x_end, y_end, angle + random.randint(15, 30), depth - 1, length * 0.7, opacity)

def generate_fractal(draw, width, height, opacity):
    """Generate a fractal tree layer."""
    x = width // 2
    y = height - 10
    generate_fractal_tree(draw, x, y, -90, depth=5, length=height // 4, opacity=opacity)


def generate_wave(draw, width, height, opacity):
    """Generate a wave pattern layer."""
    frequency = random.uniform(0.01, 0.05)
    amplitude = random.randint(20, 100)
    for y in range(height):
        for x in range(width):
            wave = int(amplitude * math.sin(frequency * x + y))
            red = (wave + 255) // 2
            green = (255 - wave) // 2
            blue = random.randint(0, 255)
            draw.point((x, y), fill=(red, green, blue, opacity))


def generate_image(width, height):
    """
    Generates a visually unique image using multiple transparent layers 
    with diverse procedural generation routines, and saves it to the 'out' directory.
    """
    try:
        start_time = time.time()  # Start timing

        # Ensure the 'out' directory exists
        output_dir = "out"
        os.makedirs(output_dir, exist_ok=True)

        # Generate a filename based on the current timestamp
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

        # Create a base image
        base_image = Image.new("RGBA", (width, height), (0, 0, 0, 255))

        # Define procedural generation routines
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

        # Generate multiple layers
        num_layers = random.randint(3, 6)  # Randomize the number of layers
        selected_routines = []
        layer_opacities = []
        for _ in range(num_layers):
            # Create a new transparent layer
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)

            # Randomly select a procedural routine
            routine_name = random.choice(list(routines.keys()))
            routine = routines[routine_name]
            selected_routines.append(routine_name)

            # Execute the routine
            opacity = random.randint(50, 200)  # Randomize layer opacity
            layer_opacities.append(opacity)
            routine(draw, width, height, opacity)

            # Blend the layer onto the base image
            base_image = Image.alpha_composite(base_image, layer)

        # Convert the final image to RGB and save it
        final_image = base_image.convert("RGB")
        final_image.save(output_path)

        end_time = time.time()  # End timing
        duration = end_time - start_time  # Calculate duration
        print(f"Image generated in {duration:.2f} seconds.")  # Log the time taken

        # Return the output path and parameters
        return output_path, {
            "routines": selected_routines,
            "opacity": layer_opacities,
            "num_layers": num_layers,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to generate image: {e}")