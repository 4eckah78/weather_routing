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


def pixel_to_flat_hex(x, y, a):
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


def doubleheight_to_pixel(row, col, a):
    # return center of hexagon
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
        pols = [list(map(float, pol.rstrip().split(','))) for pol in f.readlines()]
    return pols


def show_polygons(pols, width, height, save_to=None):
    image = Image.new("1", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    for pol in pols:
        draw.polygon((pol), fill=1)
    image.show()
    if save_to:
        image.save(save_to)


def from_pixel_to_hex_polygons(pols, map, a):
    hex_pols = []
    for polygon in pols:
        hex_pol = []
        for i in range(0, len(polygon), 2):
            row, col = pixel_to_flat_hex(polygon[i], polygon[i + 1], a)
            map[row, col // 2] = 1
            hex_pol.append((row, col))
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


def bresenham(start, end, min_max_col, map):
    double_dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(double_dx) <= abs(dy):
        return v_biased_lines(start, end, double_dx, dy, min_max_col, map)
    else:
        return h_biased_lines(start, end, double_dx, dy, min_max_col, map)


def update_min_max(min_max_col, col):
    if min_max_col[0] > col:
        min_max_col[0] = col
    if min_max_col[1] < col:
        min_max_col[1] = col


def h_biased_lines(start, end, double_dx, dy, min_max_col, map):
    row, col = start
    update_min_max(min_max_col, col)

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
        update_min_max(min_max_col, col)
    return map


def v_biased_lines(start, end, double_dx, dy, min_max_col, map):
    row, col = start
    update_min_max(min_max_col, col)

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
        update_min_max(min_max_col, col)
    return map


def get_image_size_by_polygons(pols, a):
    width, height = 0, 0
    for pol in pols:
        w, h = max(pol[::2]), max(pol[1::2])
        if w > width:
            width = w
        if h > height:
            height = h
    return int(width + 2 * a), int(height + 2 * a)


def fill_line(up_hex, down_hex, map):
    row, col = up_hex
    while (row, col) != down_hex:
        map[row, col // 2] = 1
        row, col = neighbor(row, col, "DOWN")
    if (row, col) == down_hex:
        map[row, col // 2] = 1


def fill_pol(lines, map):
    for _, line in lines.items():
        for i in range(0, len(line), 2):
            fill_line(line[i], line[i + 1], map)
    return map


def get_lines_to_fill_pol(hex_pol, min_max_col, a):
    min_c, max_c = min_max_col
    lines = {c: [] for c in range(min_c, max_c + 1)}
    for edge_id in range(len(hex_pol) - 1):
        x1, y1 = doubleheight_to_pixel(*hex_pol[edge_id], a)
        x2, y2 = doubleheight_to_pixel(*hex_pol[edge_id + 1], a)
        if x1 != x2:

            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1

            x, y = x1, y1
            dy = (y2 - y1) / (x2 - x1)

            last_col = pixel_to_flat_hex(x2, y2, a)[1]

            row, col = pixel_to_flat_hex(x, y, a)
            while col < last_col:
                x += 1.5 * a
                y += dy * 1.5 * a
                row, col = pixel_to_flat_hex(x, y, a)
                lines[col].append((row, col))

    for k, v in lines.items():
        lines[k] = sorted(lines[k])

    return lines


def raster_hex_polygons(hex_pols, map, a):
    for pol in hex_pols:
        min_max_col = [len(map[1]) * 2 + 1, -1]
        for edge_id in range(len(pol) - 1):
            map = bresenham(pol[edge_id], pol[edge_id + 1], min_max_col, map)
        lines = get_lines_to_fill_pol(pol, min_max_col, a)
        map = fill_pol(lines, map)
    return map


def draw_hex_image(need_to_draw, colour, width, height, hex_width, hex_height, dynamic_hex_map, a,
                   draw_start_end_pixels=None, polygons=None, save_to=None, show=True):
    image = Image.new("RGB", (width + scale_down_right_corner, height + scale_down_right_corner), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    for j in range(hex_height):
        for i in range(hex_width):
            x0 = 3 * a * i
            if j % 2 != 0:
                x0 = 3 * a * (2 * i + 1) / 2
            y0 = j * np.sqrt(3) * a / 2
            row, col = pixel_to_flat_hex(x0, y0, a)
            color = (255, 255, 255)
            num = dynamic_hex_map[row, col // 2]
            if num == 1:
                color = (0, 0, 0)
            if (row, col) in need_to_draw:
                color = colour
            if (row, col) in need_to_draw and num == 1:
                color = "darkred"
            draw.regular_polygon((x0, y0, a), 6, fill=color, outline=(0, 0, 0))
    if draw_start_end_pixels:
        start, end = draw_start_end_pixels
        r = 3
        draw.ellipse([start[0] - r, start[1] - r, start[0] + r, start[1] + r], fill="green")
        draw.ellipse([end[0] - r, end[1] - r, end[0] + r, end[1] + r], fill="pink")
        draw.line(draw_start_end_pixels, fill="yellow", width=2)

    if SHOW_SQUARE_AND_HEX_RASTER:
        if not polygons:
            print("No polygons in draw_hex_image arguments!")
        else:
            for pol in polygons:
                draw.polygon((pol), outline=(255, 0, 0))
    if show:
        image.show()
    if save_to:
        image.save(save_to)


def move(visited, dynamic_hex_map, end):
    for _ in range(distance):
        new_visited = set()
        for row, col in visited[-1]:
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
    for curr_layer in visited[::-1]:
        next_layer = set()
        for row, col in paths[-1]:
            neighbors = set(get_all_neighbors(row, col, dynamic_hex_map))
            next_layer = next_layer.union(set.intersection(neighbors, curr_layer))
        paths.append(next_layer)
    return paths


def get_one_path(paths, start, map):
    path = [start]
    for layer in paths[-2::-1]:
        row, col = path[-1]
        neighbors = get_all_neighbors(row, col, map)
        for r, c in neighbors:
            if (r, c) in layer:
                path.append((r, c))
                break
    return path


def scale_polygons_coords_up_left_corner(pols, scale_x, scale_y):
    new_pols = []
    for pol in pols:
        new_pols.append([(x + scale_x if j % 2 == 0 else x + scale_y) for j, x in enumerate(pol)])
    return new_pols


def rotate_polygons_around_center(polygons, start_end, center, alpha):
    new_polygons_coords = []
    scale_x, scale_y = 0, 0
    pols_start_end = polygons.copy()
    pols_start_end.append(start_end)
    for polygon in pols_start_end:
        new_coords = []
        for point in zip(polygon[::2], polygon[1::2]):
            x1, y1 = rotate_pixel_around_center(point, center, alpha)
            if x1 < scale_x:
                scale_x = x1
            if y1 < scale_y:
                scale_y = y1
            new_coords.append(x1)
            new_coords.append(y1)
        new_polygons_coords.append(new_coords)
    scale_x, scale_y = scale_up_left_corner + abs(scale_x), scale_up_left_corner + abs(scale_y)
    # print(f"scale_x={scale_x}, scale_y={scale_y}")
    new_polygons_coords = scale_polygons_coords_up_left_corner(new_polygons_coords, scale_x, scale_y)
    new_start = [new_polygons_coords[-1][0], new_polygons_coords[-1][1]]
    new_end = [new_polygons_coords[-1][2], new_polygons_coords[-1][3]]
    return new_polygons_coords, new_start, new_end


def rotate_pixel_around_center(point, center, alpha):
    x, y = point
    x0, y0 = center
    alpha_rad = np.deg2rad(alpha)
    x1 = (x - x0) * np.cos(alpha_rad) + (y - y0) * np.sin(alpha_rad)
    y1 = -(x - x0) * np.sin(alpha_rad) + (y - y0) * np.cos(alpha_rad)
    return round(x1), round(y1)


def rotate_polygons(polygons, alpha, start, end, width, height, a):
    start_end = [start[0], start[1], end[0], end[1]]
    center = [height // 2, width // 2]
    rotated_polygons, start, end = rotate_polygons_around_center(polygons, start_end, center, alpha)
    draw_start_pixel = tuple(start)
    draw_end_pixel = tuple(end)
    hex_start = pixel_to_flat_hex(start[0], start[1], a)
    hex_end = pixel_to_flat_hex(end[0], end[1], a)
    rotated_polygons = rotated_polygons[:-1]
    return rotated_polygons, hex_start, hex_end


def update_map(file, map, a, start, end, width, height, alpha=0):
    polygons = read_polygons_from(file)
    polygons = scale_polygons_coords_up_left_corner(polygons, scale_up_left_corner, scale_up_left_corner)
    if alpha:
        polygons, _, _ = rotate_polygons(polygons, alpha, start, end, width, height, a)
    hex_polygons, map = from_pixel_to_hex_polygons(polygons, map, a)
    map = raster_hex_polygons(hex_polygons, map, a)
    return map


def scale_coords(old_range_x, old_range_y, new_range_x, new_range_y, point):
    a_old, b_old = old_range_x
    a_new, b_new = new_range_x
    k_x = (b_new - a_new) / (b_old - a_old)
    x_mid_old = (b_old - a_old) / 2 + a_old
    x_mid_new = (b_new - a_new) / 2 + a_new
    a_old, b_old = old_range_y
    a_new, b_new = new_range_y
    k_y = (b_new - a_new) / (b_old - a_old)
    y_mid_old = (b_old - a_old) / 2 + a_old
    y_mid_new = (b_new - a_new) / 2 + a_new
    x_old, y_old = point
    return [(x_old - x_mid_old) * k_x + x_mid_new,
            (y_old - y_mid_old) * k_y + y_mid_new]


def create_broadcasts_from_folder(path, save_path, width, height):
    c_count = 0
    p_count = 0
    b_count = 1
    max_x = 0
    min_y = 100
    min_x = 100
    max_y = -100
    with open(path + "/w_dates.txt", "r") as fdates, \
            open(path + "/w_dims.txt", "r") as fdims, open(path + "/w_points.txt", "r") as fpoints:
        dates = fdates.readlines()
        n = len(dates)
        contours = fdims.readlines()
        points = fpoints.readlines()
        for date_line in dates[:n]:
            contours_num = int(date_line.split('\t')[1])
            with open(save_path + f"/broadcast{b_count}.txt", "w") as save_file:
                b_count += 1
                for dims_line in contours[c_count:c_count + contours_num]:
                    points_num, is_closed = list(map(int, dims_line.split(',')))
                    polygon = []
                    for point_line in points[p_count:p_count + points_num]:
                        x, y = list(map(float, point_line.strip().split(',')))
                        # y = - y
                        # x, y = scale_coords([25, 60], [9.5, 80.5], [10, width - 10], [10, height - 10], [x, y])
                        x, y = scale_coords([25, 60], [-80.5, -9.5], [10, width - 10], [10, height - 10], [x, y])
                        polygon.append(x)
                        polygon.append(y)
                        if x > max_x:
                            max_x = x
                        if y < min_y:
                            min_y = y
                        if x < min_x:
                            min_x = x
                        if y > max_y:
                            max_y = y
                    p_count += points_num
                    if not is_closed:
                        polygon.append(polygon[0])
                        polygon.append(polygon[1])
                    save_file.write(",".join([str(coord) for coord in polygon]))
                    save_file.write("\n")
            c_count += contours_num
    print(f"max_x={max_x}, min_x={min_x}, max_y={max_y}, min_y={min_y}")


# a = 20
# scaling in pixels
scale_up_left_corner = 200
scale_down_right_corner = 200
distance = 10
CLOUDS = False
SHOW_POLYGONS = False
USE_FIXED_SIZES = False
SHOW_SQUARE_AND_HEX_RASTER = True
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

    hex_polygons, static_hex_map = from_pixel_to_hex_polygons(polygons, static_hex_map, a)

    static_hex_map = raster_hex_polygons(hex_polygons, static_hex_map, a)

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
            dynamic_hex_map = update_map(f"txt/broadcast{broadcast}.txt", dynamic_hex_map, a)
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
