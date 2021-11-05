import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math


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


def generate_N_polygons(N, r_range, sides_range, show=False, save_to=None):
    # np.random.seed(1)
    sides_range[1] += 2
    image = Image.new("1", (width, height), color=0)
    draw = ImageDraw.Draw(image)

    pols = []
    for _ in range(N):
        x0 = int(np.random.rand() * width)
        y0 = int(np.random.rand() * height)
        x, y = generate_polygon(x0, y0, r_range, sides_range)
        points = list(zip(x, y))
        draw.polygon((points), fill=1)
        points = ",".join([str(x) + "," + str(y) for x, y in zip(x, y)])
        pols.append(points + '\n')
    if show:
        image.show()
    if save_to:
        with open(save_to, 'w') as output_file:
            for pol in pols:
                output_file.write(pol)


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


def show_polygons(pols, save_to=None):
    image = Image.new("1", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    for pol in pols:
        draw.polygon((pol), fill=1)
    image.show()
    if save_to:
        image.save(save_to)


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


def neighbor(row, col, direction):
    dir = doubleheight_directions[direction]
    return row + dir[0], col + dir[1]


def get_all_neighbors(row, col):
    neighbors = []
    for dir in directions:
        row1, col1 = neighbor(row, col, dir)
        if 0 <= row1 < len(hex_map) and 0 <= col1 // 2 < len(hex_map[0]):
            neighbors.append((row1, col1))
    return neighbors


def bresenham(start, end, pol_sides_list):
    double_dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(double_dx) <= abs(dy):
        return v_biased_lines(start, end, double_dx, dy, pol_sides_list)
    else:
        return h_biased_lines(start, end, double_dx, dy, pol_sides_list)


def h_biased_lines(start, end, double_dx, dy, pol_sides_list):
    row, col = start
    row1, col1 = end
    e = 0
    x_sign = 1 if double_dx >= 0 else -1
    y_sign = 1 if dy >= 0 else -1
    by = 3 * abs(dy)
    bx = 3 * abs(double_dx)
    while (row, col) != (row1, col1):
        e += by
        if e > abs(double_dx):
            row, col = neighbor(row, col, LINE_DIRS[x_sign, y_sign])
            e -= bx
        else:
            row, col = neighbor(row, col, LINE_DIRS[x_sign, 0])
            e += by
        hex_map[row, col // 2] = 1
        if row + col in pol_sides_list:
            pol_sides_list[row + col].append([row, col])
        else:
            pol_sides_list[row + col] = [[row, col]]


def v_biased_lines(start, end, double_dx, dy, pol_sides_list):
    row, col = start
    row1, col1 = end
    e = 0
    x_sign = 1 if double_dx >= 0 else -1
    y_sign = 1 if dy >= 0 else -1
    by = abs(dy)
    bx = abs(double_dx)
    while (row, col) != (row1, col1):
        e += bx
        if e > 0:
            row, col = neighbor(row, col, LINE_DIRS[x_sign, y_sign])
            e -= by
        else:
            row, col = neighbor(row, col, LINE_DIRS[-x_sign, y_sign])
            e += by
        hex_map[row, col // 2] = 1
        if row + col in pol_sides_list:
            pol_sides_list[row + col].append([row, col])
        else:
            pol_sides_list[row + col] = [[row, col]]


def get_image_size_by_polygons(pols):
    width, height = 0, 0
    for pol in pols:
        w, h = max(pol[::2]), max(pol[1::2])
        if w > width:
            width = w
        if h > height:
            height = h
    return width + 2 * a, height + 2 * a


def fill_pol(pol_sides_list):
    for summ, hexes in pol_sides_list.items():
        start_hex = min(hexes, key=lambda h: h[0])
        end_hex = max(hexes, key=lambda h: h[0])
        row, col = start_hex
        while [row, col] != end_hex:
            hex_map[row, col // 2] = 1
            row, col = neighbor(row, col, "DOWN_LEFT")


def raster_hex_polygons(hex_pols):
    for pol in hex_pols:
        pol_sides_list = {}
        for i in range(len(pol) - 1):
            bresenham(pol[i], pol[i + 1], pol_sides_list)
        fill_pol(pol_sides_list)


def draw_hex_image(save_to=None, show_in_rect=False):
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    for j in range(hex_height):
        for i in range(hex_width):
            x0 = 3 * a * i
            if j % 2 != 0:
                x0 = 3 * a * (2 * i + 1) / 2
            y0 = j * np.sqrt(3) * a / 2
            row, col = pixel_to_flat_hex(x0, y0)
            color = (255, 255, 255)
            num = hex_map[row, col // 2]
            if num == 1:
                color = (0, 0, 0)
            elif num > 1 or num == -1:
                color = "red" if num == -1 else colors[(num // distance) % len(colors)]
            draw.regular_polygon((x0, y0, a), 6, fill=color, outline=(0, 0, 0))
    if show_in_rect:
        for pol in polygons:
            draw.polygon((pol), outline=(255, 255, 255))
    image.show()
    if save_to:
        image.save(save_to)


def move(visited, fringes, t):
    for _ in range(distance):
        fringes.append([])
        for hex in fringes[-2]:
            for row, col in get_all_neighbors(hex[0], hex[1]):
                if (row, col) not in visited and hex_map[row, col // 2] != 1:
                    if (row, col // 2) == end:
                        fringes.pop()
                        move_back(t - 1, (row, col))
                        return visited, t, True
                    visited.add((row, col))
                    hex_map[row, col // 2] = t
                    fringes[-1].append((row, col))
        t += 1
    return visited, t, False


def move_back(t, end_hex):
    to_visit = [end_hex]
    next_to_visit = []
    while to_visit:

        curr = to_visit.pop()
        for row, col in get_all_neighbors(curr[0], curr[1]):
            if hex_map[row, col // 2] == t:
                hex_map[row, col // 2] = -1
                next_to_visit.append((row, col))
        if not to_visit:
            to_visit = next_to_visit
            next_to_visit = []
            t -= 1


if __name__ == "__main__":
    a = 3
    polygons_file = 'polygons.txt'

    doubleheight_directions = {
        "UP": [-2, 0],
        "DOWN": [2, 0],
        "DOWN_LEFT": [1, -1],
        "UP_LEFT": [-1, -1],
        "UP_RIGHT": [-1, 1],
        "DOWN_RIGHT": [1, 1]
    }

    directions = ("UP", "DOWN", "DOWN_LEFT", "UP_LEFT", "UP_RIGHT", "DOWN_RIGHT")

    LINE_DIRS = {
        (1, 1): "DOWN_RIGHT",
        (1, 0): "DOWN",
        (1, -1): "DOWN_LEFT",
        (-1, 0): "UP",
        (-1, 1): "UP_RIGHT",
        (-1, -1): "UP_LEFT"

    }

    polygons = read_polygons_from(polygons_file)

    width, height = get_image_size_by_polygons(polygons)

    # show_polygons(polygons)

    hex_width = math.ceil(width / (3 * a))
    hex_height = math.ceil(height / (np.sqrt(3) * a)) * 2 + 1
    hex_map = np.zeros((hex_height, hex_width), dtype=np.int32)

    hex_polygons = from_pixel_to_hex_polygons(polygons)

    raster_hex_polygons(hex_polygons)

    # import time
    # start = time.time()
    start = (0, 0)
    visited = {start}
    # 118 28
    to = (1, 1)
    end = (len(hex_map) - to[0], len(hex_map[0]) - to[1])
    fringes = [[(0, 0)]]
    t = 3
    colors = ["yellow", "purple", "green", "pink", "gray", "blue", "lemonchiffon", "lime", "orange", "salmon",
              "silver", "teal", "violet", "wheat", "yellowgreen"]
    hex_map[start] = 2
    hex_map[end] = -1
    finished = False
    distance = 10
    while not finished:
        visited, t, finished = move(visited, fringes, t)
    draw_hex_image(save_to="new.png")
    print(t)

    # for i in range(10, 15):
    #     generate_N_polygons(i, [10, 100], [3, 6], show=True, save_to=f'polygons_sample{i - 9}.txt')
    # on average 3 heptagon (polygon with 7 sides) per 100,000 cases
    # example: {5: 139255, 4: 604638, 3: 251899, 6: 4204, 7: 4} (sides_range = [3, 6]
