from PIL import Image, ImageDraw


# create a new image with a white background
img = Image.new("L", (128, 128), "white")
# create a new draw object
draw = ImageDraw.Draw(img)

# set the radius and spacing of the dots
radius = 2
spacing = 6

# loop through the image and draw the dots
for x in range(radius+1, img.width, spacing):
    for y in range(radius+1, img.height, spacing):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")

# save the image
import os
print(os.getcwd())
img.show()
img.save("Project0/data/patterns/dotted.jpeg")