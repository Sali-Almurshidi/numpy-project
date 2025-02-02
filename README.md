# Image Transformation and Composition Project

This project demonstrates various image manipulation techniques using Python, including creating tiled images, applying transformations, and generating patterns.<br> The project uses libraries like PIL, NumPy, and Matplotlib.

## **Requirements**
- Python 3.x
- `Pillow`
- `NumPy`
- `Matplotlib`

The environment setup is handled by the manipulations_env.yml file.

## **Outputs**
1. Original Image<br>
Displays the original image loaded from the file.
2. Tiled Image<br>
Creates a grid of 4 rows and 8 columns using the original image.
3. Tiled Image with Reflection<br>
Similar to the tiled image, but every alternate row is reflected horizontally.
4. Colorful Composition<br>
Arranges the image in a colorful composition using red, green, and blue color channels.
5. Collage of Transformations<br>
Creates a collage of images with various transformations, arranged in five rows.
6. Flower Pattern<br>
Displays a circular flower pattern where petals are gradient-colored versions of the image arranged around a circular center.

## **Project Structure**
- **ImageProcessor Class**
This class handles image loading and processing.
- **MainProject Class**
The MainProject class coordinates the execution of the project by calling various methods from ImageProcessor.
- **Functions**
- display_image(): Displays an image using Matplotlib.
- create_tiled_image(): Tiles the image across rows and columns.
- create_tiled_image_with_reflection(): Tiles the image with reflected rows.
- extract_color_channel(): Extracts a specific color channel (red, green, or blue) from the image.
- resize_image(): Resizes the image by a scale factor.
- create_colorful_composition(): Creates a composition with multiple color variations of the image.
- circular_crop(): Crops the image into a circle.
- apply_color_gradient(): Adds a gradient overlay to an image.
- create_flower_pattern(): Creates a flower-like pattern with gradient petals.
- apply_transformation(): Applies various transformations such as rotation, grayscale, and blur.
- transformed_images(): Generates transformed versions of the image for the collage.
- create_collage_layout(): Arranges transformed images into a collage layout.

## **Setup Instructions**
git clone <https://github.com/Sali-Almurshidi/numpy-project.git><br>
cd <numpy-project>

## **How to Run the Project**
Create the same environment by running
<pre>
conda env create -f manipulations_env.yml
</pre>
Ensure your environment is activated
<pre>
conda activate manipulations_env
</pre>
Run the main script:
<pre>
python Manipulations.py
</pre>




