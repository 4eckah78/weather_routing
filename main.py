import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

height = 480
width = 640
a = 50


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
        if 0 <= new_x <= width:
            x.append(new_x)
        else:
            x.append(width) if 0 <= new_x else x.append(0)
        if 0 <= new_y <= height:
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
            x0 = np.random.rand() * width
            y0 = np.random.rand() * height
            x, y = generate_polygon(x0, y0, r_range, sides_range)
            points = list(zip(x, y))
            draw.polygon((points), fill=1)
            points = ",".join([str(x) + "," + str(y) for x, y in zip(x, y)])
            output_file.write(points + '\n')
    # image.show()


def pixel_to_flat_hex(x, y):
    col = x * 2 / 3 / a
    row = y * 2 / np.sqrt(3) / a
    col, row = cube_to_doubleheight(*cube_round(*doubleheight_to_cube(col, row)))
    return col, row


def doubleheight_to_cube(col, row):
    x = col
    z = (row - col) / 2
    y = -x - z
    return x, y, z


def cube_to_doubleheight(x, y, z):
    col = x
    row = 2 * z + x
    return col, row


def doubleheight_to_pixel(col, row):
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


if __name__ == "__main__":
    hex_width = int(width / (2 * a)) + 100
    hex_height = int(height / (np.sqrt(3) * a)) + 100
    hex_map = np.zeros((hex_height, hex_width))
    print(hex_map.shape)
    with open("polygons.txt", 'r') as f:
        polygons = [list(map(float, pol.rstrip().split(','))) for pol in f.readlines()]

    hex_polygons = []
    for polygon in polygons:
        list = []
        for i in range(0, len(polygon), 2):
            col, row = pixel_to_flat_hex(polygon[i], polygon[i + 1])
            list.append(col)
            list.append(row)
            # print(f'hex({col}, {row}) --> hex_map[{row}, {col // 2}]')
            hex_map[row, col // 2] = 1
            x, y = doubleheight_to_pixel(col, row)
            hex_polygons.append(list)
            print(int(polygon[i]), int(polygon[i+1]), " --> ", col, row)
            print(col, row, " --> ", x, y)
        hex_polygons.append(list)

    print(hex_map)

    image = Image.new("1", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    i = 0
    color = 1
    for x in range(0, width + a, int(3 * a / 2)):
        for y in range(0, height + a, int(np.sqrt(3) * a)):
            col, row = pixel_to_flat_hex(x, y)
            if hex_map[row, col // 2] == 1:
                color = 0
            if i % 2 == 0:
                draw.regular_polygon((x, y, a), 6, fill=color, outline=0)
            else:
                draw.regular_polygon((x, y + np.sqrt(3) * a / 2, a), 6, fill=color, outline=0)
            color = 1

        i += 1
    # for polygon in hex_polygons:
    #     for i in range(0, len(polygon), 2):
    #         x, y = doubleheight_to_pixel(col, row)
    #
    #         draw.regular_polygon((x, y, a), 6, fill=0)

    image.show()

    # generate_N_polygons(10, [10, 100], [3, 6])
    # on average 3 heptagon (polygon with 7 sides) per 100,000 cases
    # example: {5: 139255, 4: 604638, 3: 251899, 6: 4204, 7: 4} (sides_range = [3, 6]
