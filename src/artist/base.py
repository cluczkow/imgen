import os
import random
import time
from PIL import Image
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

class Artist:
    """A class responsible for generating images using procedural routines."""

    def __init__(self):
        """Initialize the Artist with available procedural routines."""
        self.routines = {
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

    def generate(self, width, height):
        """
        Handle file I/O, timing, and call the generate_image method to create the image.
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

            # Call the generate_image method to create the image
            final_image, parameters = self.generate_image(width, height)

            # Save the final image
            final_image.save(output_path)

            end_time = time.time()  # End timing
            duration = end_time - start_time  # Calculate duration
            print(f"Image generated in {duration:.2f} seconds.")  # Log the time taken

            # Return the output path and parameters
            return output_path, parameters
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {e}")

    def generate_image(self, width, height):
        """
        Generate a solid color image. Subclasses can override this method to customize behavior.
        """
        print("Generating a solid color image with Artist...")

        # Generate a random color
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)

        # Create a solid color image
        base_image = Image.new("RGB", (width, height), (red, green, blue))

        # Return the final image and the parameters used
        return base_image, {
            "color": (red, green, blue),
        }