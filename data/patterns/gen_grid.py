from PIL import Image, ImageDraw

# Define image size and grid size
image_size = (128, 128)
grid_size = 6

# Create new image with white background
img = Image.new('L', image_size, color='white')

# Draw vertical grid lines
draw = ImageDraw.Draw(img)
for x in range(grid_size, 128, grid_size):
    draw.line((x, 0, x, 128), fill='black')

# Draw horizontal grid lines
for y in range(grid_size, 128, grid_size):
    draw.line((0, y, 128, y), fill='black')

# Display or save image
img.show()
img.save('Project0/data/patterns/grid.jpeg')