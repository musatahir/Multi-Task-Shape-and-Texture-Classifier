from PIL import Image, ImageDraw

# Define the size and color of the image
image_size = (128, 128)

# Create a new image and a draw object
image = Image.new("L", image_size, "white")
draw = ImageDraw.Draw(image)

# Define the width and color of the stripes
stripe_width = 2

# Draw the stripes
for x in range(0, image_size[0], stripe_width*3):
    draw.rectangle((x, 0, x+stripe_width, image_size[1]), fill=0)

# Show the image
image.show()
image.save(f"Project0/data/patterns/striped.jpeg")
