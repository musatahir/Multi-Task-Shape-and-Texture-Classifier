from PIL import Image, ImageDraw

# Define the size and color of the image
image_size = (128, 128)

# Create a new image and a draw object
image = Image.new("L", image_size, "white")
draw = ImageDraw.Draw(image)

# Define the width and color of the stripes
stripe_width = 5
stripe_color = (0, 0, 0)

# define the checkerboard pattern
for i in range(0, image_size[0], stripe_width*2):
    for j in range(0, image_size[1], stripe_width*2):
        draw.rectangle([(i, j), (i+stripe_width, j+stripe_width)], fill=0)
        draw.rectangle([(i+stripe_width, j+stripe_width), (i+stripe_width*2, j+stripe_width*2)], fill=0)
image2 = Image.new("L", image_size, "black")
image2.save("Project0/data/patterns/full.jpeg")
image.show()
image.save("Project0/data/patterns/checkerboard.jpeg")
