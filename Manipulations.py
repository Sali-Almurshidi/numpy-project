import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import math
import random

class ImageProcessor:
    def __init__(self, image_path):
        # Initialize the processor by loading and converting the image to RGBA format
        self.image = Image.open(image_path).convert('RGBA')

    # Function to display an image
    def display_image(self, image=None, title="Image", size=(10, 10)):
        # If no image is provided, display the original image
        if image is None:
            image = self.image
        plt.figure(figsize=size)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Function to create a tiled image without reflection
    def create_tiled_image(self, rows, cols):
        # Convert the image to a NumPy array (RGB format)
        img_np = np.array(self.image.convert('RGB'))
        h, w, _ = img_np.shape
        tiled_img = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Tile the image across rows and columns
        for i in range(rows):
            for j in range(cols):
                y_start, y_end = i * h, (i + 1) * h
                x_start, x_end = j * w, (j + 1) * w
                tiled_img[y_start:y_end, x_start:x_end, :] = img_np

        return Image.fromarray(tiled_img)

    # Function to create a tiled image with reflection in alternating rows
    def create_tiled_image_with_reflection(self, rows, cols):
        img_np = np.array(self.image.convert('RGB'))
        h, w, _ = img_np.shape
        tiled_img = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                y_start, y_end = i * h, (i + 1) * h
                x_start, x_end = j * w, (j + 1) * w
                # Reflect the image horizontally for odd rows
                if i % 2 == 1:
                    tiled_img[y_start:y_end, x_start:x_end, :] = np.fliplr(img_np)
                else:
                    tiled_img[y_start:y_end, x_start:x_end, :] = img_np

        return Image.fromarray(tiled_img)

    # Function to extract a color channel from the image
    def extract_color_channel(self, image, channel):
        image_color = image.copy()
        # Remove the alpha channel if present
        if image_color.shape[2] == 4:
            image_color = image_color[:, :, :3]

        mask = np.ones_like(image_color)
        mask[:, :, [1, 2] if channel == 0 else [0, 2] if channel == 1 else [0, 1]] = 0
        return image_color * mask

    # Function to resize an image by a scale factor
    def resize_image(self, image, scale_factor):
        # Convert to RGB if the image is RGBA
        if image.shape[-1] == 4:
            # Create a new array with only RGB channels
            image = image[:, :, :3]
        h, w = image.shape[:2]
        resized = np.zeros((h * scale_factor, w * scale_factor, 3), dtype=image.dtype)

        for i in range(resized.shape[0]):
            for j in range(resized.shape[1]):
                resized[i, j] = image[i // scale_factor, j // scale_factor]

        return resized

    # Function to create a colorful composition of the image using different color channels
    def create_colorful_composition(self, color_sequence):
        img_np = np.array(self.image.convert('RGB'))

        color_channels= {
            'r': self.extract_color_channel(img_np, 0),
            'g': self.extract_color_channel(img_np, 1),
            'b': self.extract_color_channel(img_np, 2)
        }

        def create_color_image(color):
            return color_channels[color]

        enlarged_image = self.resize_image(img_np, 2)

        # Create rows of colored images
        top_row = np.concatenate([create_color_image(color) for color in color_sequence[:4]], axis=1)
        # Create middle row
        left_side = np.concatenate([create_color_image(color) for color in color_sequence[4:6]], axis=0)
        right_side = np.concatenate([create_color_image(color) for color in color_sequence[-2:]], axis=0)
        middle_row = np.concatenate([left_side, enlarged_image, right_side], axis=1)
        # Create bottom row
        bottom_row = np.concatenate([create_color_image(color) for color in color_sequence[6:10]], axis=1)
        # Combine all rows
        final_composition = np.concatenate([top_row, middle_row, bottom_row], axis=0)
        return final_composition

    # Function to create a circular crop
    def circular_crop(self, image):
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)
        circular_image = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
        circular_image.putalpha(mask)
        return circular_image

    # Function to create a color gradient overlay
    def apply_color_gradient(self, img, start_color, end_color):
        """Apply a color gradient to the image."""
        gradient = Image.new('RGBA', img.size, start_color)
        draw = ImageDraw.Draw(gradient)

        for i in range(img.height):
            ratio = i / img.height
            r = int(start_color[0] + ratio * (end_color[0] - start_color[0]))
            g = int(start_color[1] + ratio * (end_color[1] - start_color[1]))
            b = int(start_color[2] + ratio * (end_color[2] - start_color[2]))
            draw.line([(0, i), (img.width, i)], fill=(r, g, b, 128))

        return Image.alpha_composite(img, gradient)

    # Function to create a flower pattern with petals around a circular center
    def create_flower_pattern(self, num_petals=6):
        canvas_size = (500, 500)
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 255))

        # Add the center circular image
        center_circle = self.circular_crop(self.image.resize((120, 120)))
        canvas.paste(center_circle, (canvas_size[0] // 2 - 60, canvas_size[1] // 2 - 60), center_circle)

        # Add petals around the center with a gradient effect
        angle_step = 360 // num_petals
        radius = 120
        center_x = canvas_size[0] // 2
        center_y = canvas_size[1] // 2
        # Define colors for alternating petals
        colors = [
            (0, 0, 255),  # Blue
            (255, 0, 0),  # Red
            (255, 182, 193),  # Light Pink
            (255, 0, 255),  # Magenta
            (170, 51, 106),  # Dark Pink
            (220, 20, 60)  # Crimson
        ]

        for i in range(num_petals):
            # Create a new circular petal for each position
            petal = self.circular_crop(self.image.resize((120, 120)))

            # Apply gradient to the petal image
            color_idx = i % len(colors)
            gradient_petal = self.apply_color_gradient(petal, colors[color_idx], (255, 255, 255))

            # Rotate the petal image
            rotated_petal = gradient_petal.rotate(i * angle_step, expand=True)

            # Calculate the petal position
            angle = math.radians(i * angle_step)
            petal_x = center_x + int(radius * math.cos(angle)) - rotated_petal.width // 2
            petal_y = center_y + int(radius * math.sin(angle)) - rotated_petal.height // 2

            canvas.paste(rotated_petal, (petal_x, petal_y), rotated_petal)


        return canvas

    # Function to apply transformations to the image
    def apply_transformation(self, transform_type):
        if transform_type == 'grayscale':
            return self.image.convert('L').convert('RGB')
        elif transform_type == 'rotate':
            return self.image.rotate(random.choice([90, 180, 270]))
        elif transform_type == 'brighten':
            enhancer = ImageEnhance.Brightness(self.image)
            return enhancer.enhance(1.5)
        elif transform_type == 'darken':
            enhancer = ImageEnhance.Brightness(self.image)
            return enhancer.enhance(0.5)
        elif transform_type == 'blur':
            return self.image.filter(ImageFilter.GaussianBlur(radius=3))
        return self.image

    def transformed_images(self):
        transformed_images = [
            self.apply_transformation('darken'),  # First row: 1 image
            self.apply_transformation('rotate'), self.apply_transformation('darken'),  # Second row: 2 identical images
            self.apply_transformation('grayscale'), self.apply_transformation('rotate'), self.apply_transformation('darken'),  # Third row: 3 images
            self.apply_transformation('brighten'),
            self.apply_transformation('grayscale'), self.apply_transformation('rotate'), self.apply_transformation('darken'),
            # Fourth row: 4 images
            self.apply_transformation('blur'), self.apply_transformation('brighten'),
            self.apply_transformation('grayscale'), self.apply_transformation('rotate'),
            self.apply_transformation('darken')  # Fifth row: 5 images
        ]
        return transformed_images

    # Create and arrange the collage
    def create_collage_layout(self, images):
        # Arrange the rows with the exact number of images
        row_layouts = [1, 2, 3, 4, 5]

        fig, axes = plt.subplots(len(row_layouts), max(row_layouts), figsize=(15, 10))

        index = 0
        for row, num_images in enumerate(row_layouts):
            for col in range(num_images):
                if index < len(images):
                    axes[row, col].imshow(images[index])
                    axes[row, col].axis('off')
                    index += 1

            # Hide empty subplots
            for col in range(num_images, max(row_layouts)):
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

class MainProject:
    def __init__(self, image_path):
        self.processor = ImageProcessor(image_path)

    def run(self):
        # Output 1: Original image
        self.processor.display_image(title="Original Image")

        # Output 2: Tiled image without reflection
        tiled_image = self.processor.create_tiled_image(rows=4, cols=8)
        self.processor.display_image(tiled_image, title="Tiled Image")

        # Output 3: Tiled image with reflection
        reflected_tiled_image = self.processor.create_tiled_image_with_reflection(rows=4, cols=6)
        self.processor.display_image(reflected_tiled_image, title="Tiled Image with Reflection")

        # Output 4: Colorful composition
        color_sequence = ['b', 'b', 'b', 'b', 'r', 'r', 'g', 'g', 'g', 'g', 'r', 'r']
        colorful_composition = self.processor.create_colorful_composition(color_sequence)
        self.processor.display_image(colorful_composition, title="Colorful Composition")

        #Output 5: Collage with transformations
        self.processor.create_collage_layout(self.processor.transformed_images())

        # Output 6: Flower pattern
        flower_pattern = self.processor.create_flower_pattern()
        self.processor.display_image(flower_pattern, title="Flower Pattern")


# Start the project
if __name__ == "__main__":
    project = MainProject('Monkey.jpeg')
    project.run()
