import math

def generate_star_points(size, center):
    points = []
    for i in range(5):
        angle = (2 * math.pi * i / 5) - math.pi / 2
        x = center[0] + size * math.cos(angle)
        y = center[1] + size * math.sin(angle)
        points.append((x, y))
        angle += math.pi / 5
        x = center[0] + size / 2 * math.cos(angle)
        y = center[1] + size / 2 * math.sin(angle)
        points.append((x, y))
    return points

