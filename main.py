import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math

height = 480
width = 640
a = 15
polygons_file = 'polygons.txt'


def generate_polygon(x0, y0, r_range, n_range):
    r = np.random.randint(r_range[0], r_range[1])
    pi = np.pi
    a = 2 * pi / n_range[1]
    b = 2 * pi / n_range[0]
    fi = np.random.rand() * (b - a) + a
    x = []
    y = []
    while fi <= 2 * pi:
        new_x = int(r * np.cos(fi)) + x0
        new_y = int(r * np.sin(fi)) + y0
        if 0 <= new_x < width:
            x.append(new_x)
        else:
            x.append(width) if 0 <= new_x else x.append(0)
        if 0 <= new_y < height:
            y.append(new_y)
        else:
            y.append(height) if 0 <= new_y else y.append(0)
        fi += np.random.rand() * (b - a) + a
    x.append(x[0])
    y.append(y[0])
    return np.asarray(x), np.asarray(y)


def generate_N_polygons(N, r_range, sides_range):
    np.random.seed(1)
    sides_range[1] += 2
    image = Image.new("1", (width, height), color=0)
    draw = ImageDraw.Draw(image)

    with open("polygons.txt", 'w') as output_file:
        for _ in range(N):
            x0 = int(np.random.rand() * width)
            y0 = int(np.random.rand() * height)
            x, y = generate_polygon(x0, y0, r_range, sides_range)
            points = list(zip(x, y))
            draw.polygon((points), fill=1)
            points = ",".join([str(x) + "," + str(y) for x, y in zip(x, y)])
            output_file.write(points + '\n')
    image.show()


def pixel_to_flat_hex(x, y):
    row = y * 2 / np.sqrt(3) / a
    col = x * 2 / 3 / a
    row, col = cube_to_doubleheight(*cube_round(*doubleheight_to_cube(row, col)))
    return row, col


def doubleheight_to_cube(row, col):
    x = col
    z = (row - col) / 2
    y = -x - z
    return x, y, z


def cube_to_doubleheight(x, y, z):
    row = 2 * z + x
    col = x
    return row, col


def doubleheight_to_pixel(row, col):
    x = a * 3 / 2 * col
    y = a * np.sqrt(3) / 2 * row
    return x, y


def cube_round(x, y, z):
    rx = round(x)
    ry = round(y)
    rz = round(z)

    x_diff = abs(rx - x)
    y_diff = abs(ry - y)
    z_diff = abs(rz - z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return rx, ry, rz


def read_polygons_from(filename):
    with open(filename, 'r') as f:
        pols = [list(map(int, pol.rstrip().split(','))) for pol in f.readlines()]
    return pols


def from_pixel_to_hex_polygons(pols):
    hex_pols = []
    for polygon in pols:
        hex_pol = []
        for i in range(0, len(polygon), 2):
            row, col = pixel_to_flat_hex(polygon[i], polygon[i + 1])
            hex_map[row, col // 2] = 1
            hex_pol.append([row, col])
            # x, y = doubleheight_to_pixel(row, col)
            # print(int(polygon[i]), int(polygon[i + 1]), " --> ", row, col)
            # print(row, col, " --> ", x, y)
        hex_pols.append(hex_pol)
    return hex_pols


def show_hex_image(width, height, hex_width, hex_height, a):
    image = Image.new("1", (width, height), color=1)
    draw = ImageDraw.Draw(image)

    for j in range(hex_height):
        for i in range(hex_width):
            x0 = 3 * a * i
            if j % 2 != 0:
                x0 = 3 * a * (2 * i + 1) / 2
            y0 = j * np.sqrt(3) * a / 2
            row, col = pixel_to_flat_hex(x0, y0)
            color = 0 if hex_map[row, col // 2] == 1 else 1
            draw.regular_polygon((x0, y0, a), 6, fill=color, outline=0)
    image.show()


if __name__ == "__main__":
    hex_width = math.ceil(width / (3 * a))
    hex_height = math.ceil(height / (np.sqrt(3) * a)) * 2 + 1
    hex_map = np.zeros((hex_height, hex_width), dtype=np.uint8)

    polygons = read_polygons_from(polygons_file)

    hex_polygons = from_pixel_to_hex_polygons(polygons)

    show_hex_image(width, height, hex_width, hex_height, a)

    # generate_N_polygons(10, [10, 100], [3, 6])
    # on average 3 heptagon (polygon with 7 sides) per 100,000 cases
    # example: {5: 139255, 4: 604638, 3: 251899, 6: 4204, 7: 4} (sides_range = [3, 6]