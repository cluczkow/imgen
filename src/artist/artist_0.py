from .base import Artist
from PIL import Image, ImageDraw
import random

class Artist_0(Artist):
    """A subclass of Artist that generates images with multiple layers and procedural routines."""

    def __init__(self):
        super().__init__()
        # You can add additional initialization logic here if needed

    def generate_image(self, width, height):
        """Generate an image with multiple layers and procedural routines."""
        print("Generating an image with Artist_0's unique style...")

        # Create a base image
        base_image = Image.new("RGBA", (width, height), (0, 0, 0, 255))

        # Generate multiple layers
        num_layers = random.randint(3, 6)  # Randomize the number of layers
        selected_routines = []
        layer_opacities = []
        for _ in range(num_layers):
            # Create a new transparent layer
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)

            # Randomly select a procedural routine
            routine_name = random.choice(list(self.routines.keys()))
            routine = self.routines[routine_name]
            selected_routines.append(routine_name)

            # Execute the routine
            opacity = random.randint(50, 200)  # Randomize layer opacity
            layer_opacities.append(opacity)
            routine(draw, width, height, opacity)

            # Blend the layer onto the base image
            base_image = Image.alpha_composite(base_image, layer)

        # Convert the final image to RGB
        final_image = base_image.convert("RGB")

        # Return the final image and the parameters used
        return final_image, {
            "routines": selected_routines,
            "opacity": layer_opacities,
            "num_layers": num_layers,
        }