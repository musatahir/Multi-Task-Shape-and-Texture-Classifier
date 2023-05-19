from PIL import Image, ImageDraw, ImageFilter
import random
from helpers import generate_star_points
import numpy as np
import os

# Define image size 
image_size = (128, 128)
images_dataset_path = "data/saved-images"
shapes = ['circle', 'square', 'triangle', 'rectangle', 'star']
patterns = ['full','striped','checkerboard','dotted','grid']
min_size = 20
max_size = 60
shape_num = 50

if not os.path.exists(images_dataset_path):
        os.makedirs(images_dataset_path)

for shape in shapes:
    if not os.path.exists(f"{images_dataset_path}/{shape}"):
        os.makedirs(f"{images_dataset_path}/{shape}")
    for pattern in patterns:
        if not os.path.exists(f"{images_dataset_path}/{shape}/{pattern}"):
            os.makedirs(f"{images_dataset_path}/{shape}/{pattern}")
    
for shape in shapes:
    for i in range(shape_num):
        for pattern in patterns:

            # Create a grayscale image with black background
            image = Image.new('L', image_size, color = 'white')
            # Create an image draw object
            draw = ImageDraw.Draw(image)
            size = random.randint(min_size, max_size)
            x = random.randint(size + 3, image_size[1] - size - 3)
            y = random.randint(size + 3, image_size[0] - size - 3)
            
            if shape == 'circle':
                draw.ellipse((x, y, x+size, y+size), fill=0)
                p = Image.open(f"data/patterns/{pattern}.jpeg")
                image = image.filter(ImageFilter.GaussianBlur(1))
                p.paste(image, mask=image)

            elif shape == 'square':
                coords = [(x,y), (x+size,y), (x+size, y+size), (x, y+size)]
                draw.polygon(coords, fill=0)
                p = Image.open(f"data/patterns/{pattern}.jpeg")
                image = image.filter(ImageFilter.GaussianBlur(1))
                p.paste(image, mask=image)
                
                #draw.rectangle((x, y, x+size, y+size), fill=0)

            elif shape == 'triangle':
                coords = [(x + size/2, y), (x, y + size), (x + size, y + size)]
                draw.polygon(coords, fill=0)
                p = Image.open(f"data/patterns/{pattern}.jpeg")
                image = image.filter(ImageFilter.GaussianBlur(1))
                p.paste(image, mask=image)
            elif shape == 'rectangle':
                coords = [(x,y), (x+size,y), (x+size, y+size/2), (x, y+size/2)]
                draw.polygon(coords, fill=0)
                p = Image.open(f"data/patterns/{pattern}.jpeg")
                image = image.filter(ImageFilter.GaussianBlur(1))
                p.paste(image, mask=image)
                #draw.rectangle((x, y, x+size, y+size/2), fill=0)
            elif shape == 'star':
                x_star = random.randint(size+3, image_size[1] - size-3)
                y_star = random.randint(size+3, image_size[0] - size-3)
                coords = generate_star_points(size, (x_star,y_star))
                draw.polygon(coords, fill=0)
                p = Image.open(f"data/patterns/{pattern}.jpeg")
                image = image.filter(ImageFilter.GaussianBlur(1))
                p.paste(image, mask=image)
            
            p.save(f"data/saved-images/{shape}/{pattern}/{shape}_{pattern}_{i}.jpg")












