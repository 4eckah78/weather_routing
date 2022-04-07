import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import math
import copy


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


def show_polygons(pols, width, height, save_to=None):
    image = Image.new("1", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    for pol in pols:
        draw.polygon((pol), fill=1)
    image.show()
    if save_to:
        image.save(save_to)


def from_pixel_to_hex_polygons(pols, map):
    hex_pols = []
    for polygon in pols:
        hex_pol = []
        for i in range(0, len(polygon), 2):
            row, col = pixel_to_flat_hex(polygon[i], polygon[i + 1])
            map[row, col // 2] = 1
            hex_pol.append([row, col])
            # x, y = doubleheight_to_pixel(row, col)
            # print(int(polygon[i]), int(polygon[i + 1]), " --> ", row, col)
            # print(row, col, " --> ", x, y)
        hex_pols.append(hex_pol)
    return hex_pols, map


def neighbor(row, col, direction):
    dir = doubleheight_directions[direction]
    return row + dir[0], col + dir[1]


def get_all_neighbors(row, col, map):
    neighbors = []
    for dir in directions:
        row1, col1 = neighbor(row, col, dir)
        if 0 <= row1 < len(map) and 0 <= col1 // 2 < len(map[0]):
            neighbors.append((row1, col1))
    return neighbors


def bresenham(start, end, pol_sides_list, map):
    double_dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(double_dx) <= abs(dy):
        return v_biased_lines(start, end, double_dx, dy, pol_sides_list, map)
    else:
        return h_biased_lines(start, end, double_dx, dy, pol_sides_list, map)


def h_biased_lines(start, end, double_dx, dy, pol_sides_list, map):
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
        map[row, col // 2] = 1
        if row + col in pol_sides_list:
            pol_sides_list[row + col].append([row, col])
        else:
            pol_sides_list[row + col] = [[row, col]]
    return map


def v_biased_lines(start, end, double_dx, dy, pol_sides_list, map):
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
        map[row, col // 2] = 1
        if row + col in pol_sides_list:
            pol_sides_list[row + col].append([row, col])
        else:
            pol_sides_list[row + col] = [[row, col]]
    return map


def get_image_size_by_polygons(pols):
    width, height = 0, 0
    for pol in pols:
        w, h = max(pol[::2]), max(pol[1::2])
        if w > width:
            width = w
        if h > height:
            height = h
    return width + 2 * a, height + 2 * a


def fill_pol(pol_sides_list, map):
    for summ, hexes in pol_sides_list.items():
        start_hex = min(hexes, key=lambda h: h[0])
        end_hex = max(hexes, key=lambda h: h[0])
        row, col = start_hex
        while [row, col] != end_hex:
            map[row, col // 2] = 1
            row, col = neighbor(row, col, "DOWN_LEFT")
    return map


def raster_hex_polygons(hex_pols, map):
    for pol in hex_pols:
        pol_sides_list = {}
        for i in range(len(pol) - 1):
            map = bresenham(pol[i], pol[i + 1], pol_sides_list, map)
        map = fill_pol(pol_sides_list, map)
    return map


def draw_hex_image(need_to_draw, colour, width, height, hex_width, hex_height, dynamic_hex_map, polygons=None, save_to=None):
    image = Image.new("RGB", (width + scale_down_right_corner, height + scale_down_right_corner), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    for j in range(hex_height):
        for i in range(hex_width):
            x0 = 3 * a * i
            if j % 2 != 0:
                x0 = 3 * a * (2 * i + 1) / 2
            y0 = j * np.sqrt(3) * a / 2
            row, col = pixel_to_flat_hex(x0, y0)
            color = (255, 255, 255)
            num = dynamic_hex_map[row, col // 2]
            if num == 1:
                color = (0, 0, 0)
            if (row, col) in need_to_draw:
                color = colour
            if (row, col) in need_to_draw and num == 1:
                color = "darkred"
            draw.regular_polygon((x0, y0, a), 6, fill=color, outline=(0, 0, 0))
    if SHOW_SQUARE_AND_HEX_RASTER:
        if not polygons:
            print("No polygons in draw_hex_image arguments!")
        else:
            for pol in polygons:
                draw.polygon((pol), outline=(255, 0, 0))
    if SHOW:
        image.show()
    if save_to:
        image.save(save_to)


def move(visited, dynamic_hex_map, end):
    for _ in range(distance):
        to_check = copy.copy(visited[-1])
        new_visited = set()
        while to_check:
            row, col = to_check.pop()
            if dynamic_hex_map[row, col // 2] != 1:
                new_visited.add((row, col))
            for row1, col1 in get_all_neighbors(row, col, dynamic_hex_map):
                if dynamic_hex_map[row1, col1 // 2] != 1:
                    if (row1, col1) == end:
                        return True
                    new_visited.add((row1, col1))
        visited.append(new_visited)
    return False


def move_back(visited, end, dynamic_hex_map):
    paths = [set([end])]
    i = 0
    for curr_layer in visited[::-1]:
        i += 1
        next_layer = set()
        for row, col in paths[-1]:
            neighbors = set(get_all_neighbors(row, col, dynamic_hex_map))
            next_layer = next_layer.union(set.intersection(neighbors, curr_layer))
        paths.append(next_layer)
    return paths


def update_map(file, map):
    polygons = read_polygons_from(file)
    hex_polygons, map = from_pixel_to_hex_polygons(polygons, map)
    map = raster_hex_polygons(hex_polygons, map)
    return map


a = 20
# scaling in pixels
scale_up_left_corner = 10
scale_down_right_corner = 0
distance = 20
CLOUDS = False
SHOW = True
SHOW_POLYGONS = False
USE_FIXED_SIZES = True
SHOW_SQUARE_AND_HEX_RASTER = False
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

if __name__ == "__main__":
    polygons = read_polygons_from(polygons_file)

    width, height = get_image_size_by_polygons(polygons)
    if USE_FIXED_SIZES:
        width, height = 640, 480

    if SHOW_POLYGONS:
        show_polygons(polygons, width, height)

    hex_width = math.ceil(width / (3 * a))
    hex_height = math.ceil(height / (np.sqrt(3) * a) * 2) + 1
    static_hex_map = np.zeros((hex_height, hex_width), dtype=np.int32)

    hex_polygons, static_hex_map = from_pixel_to_hex_polygons(polygons, static_hex_map)

    static_hex_map = raster_hex_polygons(hex_polygons, static_hex_map)

    start = (0, 0)
    visited = [{start}]
    end_row = len(static_hex_map) - 2
    end_col = (len(static_hex_map[0]) - 1) * 2 + (1 if end_row % 2 != 0 else 0)
    end = (end_row, end_col)
    colors = ["yellow", "purple", "green", "pink", "gray", "blue", "lime", "orange", "salmon",
              "silver", "teal", "violet", "wheat"]
    max_iterations = static_hex_map.shape[0] * static_hex_map.shape[1]
    iterations = 0
    finished = False
    broadcast = 1
    colour = 0
    while not finished and iterations < max_iterations:
        if broadcast < 6:
            dynamic_hex_map = np.copy(static_hex_map)
            dynamic_hex_map = update_map(f"txt/broadcast{broadcast}.txt", dynamic_hex_map)
            broadcast += 1
        finished = move(visited, dynamic_hex_map, end)
        if finished:
            paths = move_back(visited, end, dynamic_hex_map)
            to_color = set.union(*paths)
            draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                           dynamic_hex_map)  # ,save_to=f"pictures/broadcast{8}.png")
        else:
            to_color = set.union(*visited[-distance:])
            # draw_hex_image(to_color, colors[colour % len(colors)], width, height, hex_width, hex_height, dynamic_hex_map)
            print(iterations)
        colour += 1
        iterations += 1
        if iterations >= max_iterations:
            exit(f"number of steps exceeded max_steps: {max_iterations}")

    broadcast = 1
    to_color = set([start])
    paths.pop()
    paths = paths[::-1]
    time = math.ceil(len(paths) / distance)
    # draw_hex_image({(0,0)}, "red", width, height, hex_width, hex_height, dynamic_hex_map, save_to=f"demo_dyn_2/broadcast{0}.png")
    i = 1
    if not CLOUDS:
        for step in range(0, distance * time, distance):
            layer = paths[step:step + distance]
            if broadcast < 6:
                dynamic_hex_map = np.copy(static_hex_map)
                dynamic_hex_map = update_map(f"txt/broadcast{broadcast}.txt", dynamic_hex_map)
                broadcast += 1
            draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map)
            i += 1
            new_layer = set.union(*layer)
            to_color = set.union(to_color, new_layer)
            draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map)
            i += 1
    else:
        for step in range(0, distance * time, distance):
            layer = paths[step:step + distance]
            if broadcast < 6:
                dynamic_hex_map = np.copy(static_hex_map)
                dynamic_hex_map = update_map(f"txt/broadcast{broadcast}.txt", dynamic_hex_map)
                broadcast += 1
            draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map)
            i += 1
            to_color = set.union(*layer)
            draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map)
            i += 1
