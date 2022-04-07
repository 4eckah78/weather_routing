from main import *


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
    scale_x, scale_y = 500, 500
    print(f"scale_x={scale_x}, scale_y={scale_y}")
    new_polygons_coords = scale_polygons_coords_up_left_corner(new_polygons_coords, scale_x, scale_y)
    new_start = [new_polygons_coords[-1][0], new_polygons_coords[-1][1]]
    new_end = [new_polygons_coords[-1][2], new_polygons_coords[-1][3]]
    return new_polygons_coords, new_start, new_end


def rotate_pixel_around_center(point, center, alpha):
    x, y = point
    x0, y0 = center
    x1 = (x - x0) * np.cos(np.deg2rad(alpha)) + (y - y0) * np.sin(np.deg2rad(alpha))
    y1 = -(x - x0) * np.sin(np.deg2rad(alpha)) + (y - y0) * np.cos(np.deg2rad(alpha))
    return round(x1), round(y1)


def scale_polygons_coords_up_left_corner(pols, scale_x, scale_y):
    new_pols = []
    for pol in pols:
        new_pols.append([(x + scale_x if j % 2 == 0 else x + scale_y) for j, x in enumerate(pol)])
    return new_pols



if __name__ == "__main__":
    count = 1
    for alpha in range(0, 361, 30):
        polygons = read_polygons_from(polygons_file)

        polygons = scale_polygons_coords_up_left_corner(polygons, scale_up_left_corner, scale_up_left_corner)

        width, height = get_image_size_by_polygons(polygons)
        if USE_FIXED_SIZES:
            width, height = 640, 480

        start = (scale_up_left_corner, scale_up_left_corner)
        end = (width - 1 + scale_up_left_corner, height - 1 + scale_up_left_corner)
        start_end = [start[0], start[1], end[0], end[1]]

        center = [height // 2, width // 2]
        rotated_polygons, start, end = rotate_polygons_around_center(polygons, start_end, center, alpha)
        new_width, new_height = get_image_size_by_polygons(rotated_polygons)
        rotated_polygons = rotated_polygons[:-1]
        if SHOW_POLYGONS:
            show_polygons(polygons, width, height)
            show_polygons(rotated_polygons, new_width, new_height)

        polygons = rotated_polygons
        height, width = 1000, 1000
        print(f"height={height}, width={width}")

        hex_width = math.ceil((width + scale_down_right_corner) / (3 * a))
        hex_height = math.ceil((height + scale_down_right_corner) / (np.sqrt(3) * a) * 2) + 1
        static_hex_map = np.zeros((hex_height, hex_width), dtype=np.int32)

        hex_polygons, static_hex_map = from_pixel_to_hex_polygons(polygons, static_hex_map)

        static_hex_map = raster_hex_polygons(hex_polygons, static_hex_map)

        # draw_hex_image([], "red", width, height, hex_width, hex_height,
        #                static_hex_map, polygons=polygons)  # , save_to=f"experiments_with_parallelogram/experiment7.png")

        start = pixel_to_flat_hex(start[0], start[1])

        visited = [{start}]
        end = pixel_to_flat_hex(end[0], end[1])

        # end_row = len(static_hex_map) - 2
        # end_col = (len(static_hex_map[0]) - 1) * 2 + (1 if end_row % 2 != 0 else 0)
        #
        # draw_hex_image([start, end], "red", width, height, hex_width, hex_height,
        #                static_hex_map, polygons=polygons)  # , save_to=f"experiments_with_parallelogram/experiment7.png")

        colors = ["yellow", "purple", "green", "pink", "gray", "blue", "lime", "orange", "salmon",
                  "silver", "teal", "violet", "wheat"]
        max_iterations = static_hex_map.shape[0] * static_hex_map.shape[1]
        iterations = 0
        finished = False
        broadcast = 1
        colour = 0
        while not finished and iterations < max_iterations:
            finished = move(visited, static_hex_map, end)
            if finished:
                paths = move_back(visited, end, static_hex_map)
                to_color = set.union(*paths)
                draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                               static_hex_map, save_to=f"experiments_with_rotating/experiment_{alpha}_degree_{a}_hex.png")
                count+=1
            else:
                to_color = set.union(*visited[-distance:])
                # draw_hex_image(to_color, colors[colour % len(colors)], width, height, hex_width, hex_height, dynamic_hex_map)
                print(iterations)
            colour += 1
            iterations += 1
            if iterations >= max_iterations:
                draw_hex_image([start, end], "red", width, height, hex_width, hex_height,
                               static_hex_map, polygons=polygons)  # , save_to=f"experiments_with_parallelogram/experiment7.png")
                exit(f"number of steps exceeded max_steps: {max_iterations}")

