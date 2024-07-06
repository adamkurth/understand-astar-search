from PIL import Image, ImageSequence
import os

# Directory where the search steps images are stored
output_folder = 'search_steps'
output_gif = 'step_search.gif'

# Create a list of image files in the output folder
image_files = [os.path.join(output_folder, filename) for filename in os.listdir(output_folder) if filename.startswith("step_") and filename.endswith(".png")]
image_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Create a list of image objects from the image files
images = [Image.open(image_file) for image_file in image_files]

# Convert the images to the RGB color space
images = [image.convert('RGB') for image in images]

# Save the images as an animated GIF
images[0].save(output_gif, save_all=True, append_images=images[1:], duration=2000, loop=0)

print(f'Animated GIF saved as {output_gif}')