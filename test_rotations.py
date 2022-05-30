from main import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time


def experiment(a_list, alpha_list, SHOW=False, show_dynamic=False, opt=False, broadcasts_path=None,
               static_constraints_path="polygons.txt", save_to=None, start_in_cntr=False):
    result = np.zeros((len(a_list), len(alpha_list)))
    pics = 0
    for i, a in enumerate(a_list):
        print(f"a = {a}")
        for j, alpha in enumerate(alpha_list):
            # print(f"alpha={alpha}")
            begin = time.time()
            if static_constraints_path != "polygons.txt":
                polygons = read_polygons_from(static_constraints_path + f"broadcast1.txt")
            else:
                polygons = read_polygons_from(static_constraints_path)

            pixel_width, pixel_height = get_image_size_by_polygons(polygons)
            pixel_width, pixel_height = pixel_width + int(2*71), pixel_height + int(2*71)
            if USE_FIXED_SIZES:
                pixel_width, pixel_height = 640, 480

            pixel_start = (0 + scale_up_left_corner, 0 + scale_up_left_corner)
            pixel_end = (pixel_width - 1 + scale_up_left_corner, pixel_height - 1 + scale_up_left_corner)
            length = np.sqrt((pixel_end[0]-pixel_start[0])**2 + (pixel_end[1]-pixel_start[1])**2)
            print(f"distance = {length}")

            draw_start_pixel = tuple(pixel_start)
            draw_end_pixel = tuple(pixel_end)
            # put start in the center of hex
            if start_in_cntr:
                start = pixel_to_flat_hex(pixel_start[0], pixel_start[1], a)
                new_pixel_start = doubleheight_to_pixel(*start, a)
                scale_x, scale_y = new_pixel_start[0] - pixel_start[0], new_pixel_start[1] - pixel_start[1]
                new_pixel_end = pixel_end[0] + scale_x, pixel_end[1] + scale_y
                pixel_start, pixel_end = new_pixel_start, new_pixel_end
                polygons = scale_polygons_coords_up_left_corner(polygons, scale_x, scale_y)
                draw_start_pixel = tuple(pixel_start)
                draw_end_pixel = tuple(pixel_end)


            polygons = scale_polygons_coords_up_left_corner(polygons, scale_up_left_corner, scale_up_left_corner)

            # turn to 30 degree (optimal)
            if opt:
                betta = np.rad2deg(np.arctan((pixel_end[1] - pixel_start[1]) / (pixel_end[0] - pixel_start[0])))
                opt_alpha = -30 + betta
                alpha = opt_alpha

            # start_end = [start[0], start[1], end[0], end[1]]
            # if alpha:
            #     center = [height // 2, width // 2]
            #     rotated_polygons, start, end = rotate_polygons_around_center(polygons, start_end, center, alpha)
            #     draw_start_pixel = tuple(start)
            #     draw_end_pixel = tuple(end)
            #     new_width, new_height = get_image_size_by_polygons(rotated_polygons, a)
            #     rotated_polygons = rotated_polygons[:-1]
            #     if SHOW_POLYGONS:
            #         show_polygons(polygons, width, height)
            #         show_polygons(rotated_polygons, new_width, new_height)
            #
            #     polygons = rotated_polygons
            width, height = get_image_size_by_polygons(polygons)
            width, height = width + int(2*71), height + int(2*71)
            if USE_FIXED_SIZES:
                width, height = 750, 590
            # print(f"height={height}, width={width}")

            hex_width = math.ceil((width + scale_down_right_corner) / (3 * a))
            hex_height = math.ceil((height + scale_down_right_corner) / (np.sqrt(3) * a) * 2) + 1
            # print(f"hex width x height: {hex_width}x{hex_height}")
            # print(f"width x height: {width}x{height}")
            static_hex_map = np.zeros((hex_height, hex_width), dtype=np.int32)
            # draw = pixel_to_flat_hex(polygons[0][0], polygons[0][1], a)
            # draw_hex_image([draw], "red", width, height, hex_width, hex_height, static_hex_map, a,
            #                polygons=polygons)
            hex_polygons, static_hex_map = from_pixel_to_hex_polygons(polygons, static_hex_map, a)
            static_hex_map = raster_hex_polygons(hex_polygons, static_hex_map, a)
            # draw_hex_image([start, end], "red", width, height, hex_width, hex_height, static_hex_map, a,
            #                polygons=polygons)
            start = pixel_to_flat_hex(pixel_start[0], pixel_start[1], a)
            end = pixel_to_flat_hex(pixel_end[0], pixel_end[1], a)

            colors = ["yellow", "purple", "green", "pink", "gray", "blue", "lime", "orange", "salmon",
                      "silver", "teal", "violet", "wheat"]

            max_iterations = static_hex_map.shape[0] * static_hex_map.shape[1]
            iterations = 0
            finished = False
            colour = 0
            broadcast = 1
            if SHOW:
                draw_hex_image([start, end], "red", width, height, hex_width, hex_height, static_hex_map, a, polygons=polygons)
            if save_to:
                draw_hex_image([start, end], "red", width, height, hex_width, hex_height, static_hex_map, a
                               , save_to=f"{save_to}/pic{pics}.png", show=False)
                pics += 1
            if alpha:
                polygons, start, end = rotate_polygons_start_end(polygons, alpha, pixel_start, pixel_end, pixel_width,
                                                                 pixel_height, a)
                static_hex_map = np.zeros((hex_height, hex_width), dtype=np.int32)
                hex_polygons, static_hex_map = from_pixel_to_hex_polygons(polygons, static_hex_map, a)
                static_hex_map = raster_hex_polygons(hex_polygons, static_hex_map, a)
                if SHOW:
                    draw_hex_image([start, end], "red", width, height, hex_width, hex_height, static_hex_map, a)
                if save_to:
                    draw_hex_image([start, end], "red", width, height, hex_width, hex_height, static_hex_map, a
                                   , save_to=f"{save_to}/pic{pics}.png", show=False)
                    pics += 1

            visited = [{start}]
            while not finished and iterations < max_iterations:
                dynamic_hex_map = np.copy(static_hex_map)
                if broadcasts_path:
                    try:
                        dynamic_hex_map = update_map(f"{broadcasts_path}broadcast{broadcast}.txt", dynamic_hex_map, a,
                                                 pixel_start, pixel_end, width, height, alpha=alpha)
                    except FileNotFoundError as e:
                        print(f"[ERROR] missing forecast files!!!")
                        draw_hex_image([start, end], "red", width, height, hex_width, hex_height, dynamic_hex_map, a)
                        stats.append(0)
                        real_distance.append(0)
                        break
                broadcast += 1
                # draw_hex_image([start, end], "red", width, height, hex_width, hex_height, dynamic_hex_map, a)
                bg = time.time()
                finished = move(visited, dynamic_hex_map, end)
                if finished:
                    paths = move_back(visited, end, dynamic_hex_map)
                    to_color = set.union(*paths)
                    if SHOW:
                        draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                                       dynamic_hex_map,
                                       a, draw_start_end_pixels=[draw_start_pixel, draw_end_pixel])
                    if save_to:
                        draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map, a
                                       , save_to=f"{save_to}/pic{pics}.png", show=False)
                        pics += 1
                    one_path = get_one_path(paths, start, dynamic_hex_map)
                    broadcast += 1
                    if SHOW:
                        draw_hex_image(one_path, "red", width, height, hex_width, hex_height, dynamic_hex_map, a)
                    if save_to:
                        draw_hex_image(one_path, "red", width, height, hex_width, hex_height, dynamic_hex_map, a
                                       , save_to=f"{save_to}/pic{pics}.png", show=False)
                        pics += 1
                    if show_dynamic:
                        paths = [{hex} for hex in one_path[::-1]]
                        broadcast = 1
                        to_color = set([start])
                        paths.pop()
                        paths = paths[::-1]
                        t = math.ceil(len(paths) / distance)
                        for step in range(0, distance * t, distance):
                            layer = paths[step:step + distance]
                            dynamic_hex_map = np.copy(static_hex_map)
                            if broadcasts_path:
                                dynamic_hex_map = update_map(f"{broadcasts_path}broadcast{broadcast}.txt",
                                                             dynamic_hex_map,
                                                             a, pixel_start, pixel_end, width, height, alpha=alpha)

                            broadcast += 1
                            to_color.add(end)
                            draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map, a,
                                           show=SHOW)
                            if save_to:
                                draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                                               dynamic_hex_map, a, save_to=f"{save_to}/pic{pics}.png", show=False)
                                pics += 1
                            new_layer = set.union(*layer)
                            to_color = set.union(to_color, new_layer)
                            draw_hex_image(to_color, "red", width, height, hex_width, hex_height, dynamic_hex_map, a,
                                           show=SHOW)
                            if save_to:
                                draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                                               dynamic_hex_map, a, save_to=f"{save_to}/pic{pics}.png", show=False)
                                pics += 1
                    # print(f"optimal path's length in hexes = {len(paths) - 1}")
                    stats_hex.append(len(paths) - 1)
                    path_length_in_pixels = round((len(paths) - 1) * np.sqrt(3) * a)
                    stats.append(path_length_in_pixels)
                    # print(f"optimal path's length in pixels = {path_length_in_pixels}")
                    result[i][j] = path_length_in_pixels
                    # print(f"alpha={alpha}")
                else:
                    to_color = set.union(*visited[-distance:])
                    # draw_hex_image(to_color, colors[colour % len(colors)], width, height, hex_width, hex_height, dynamic_hex_map, a)
                    #                     # print(f"iteration {iterations}")
                colour += 1
                iterations += 1
                # print(f"function move took {int(time.time() - bg)} seconds, length of visited = {len(visited[-1])}")
                if iterations >= max_iterations:
                    draw_hex_image([start, end], "red", width, height, hex_width, hex_height,
                                   static_hex_map, a,
                                   polygons=polygons)  # , save_to=f"experiments_with_parallelogram/experiment7.png")
                    print(f"[ERROR] number of steps exceeded max_steps: {max_iterations}!!!")
                    result[i][j] = 0
                    stats.append(0)
            print(f"time = {int(time.time() - begin)} seconds")
            time_stats.append(time.time() - begin)
    return result


def plot_3D(list_1, list_2, func):
    X, Y = np.meshgrid(list_2, list_1)
    Z = func(list_1, list_2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('alpha')
    ax.set_ylabel('a')
    ax.set_zlabel('length in pixels')
    ax.set_title("length_in_pixels(a, alpha)")
    plt.show()


def plot_graphic(x, y, title, x_labl, y_labl):
    plt.plot(x, y), plt.title(title)
    plt.xlabel(x_labl), plt.ylabel(y_labl)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    count = 1
    stats = []
    stats_hex = []
    time_stats = []
    a = 10
    alpha = 0
    real_distance = []
    # a_list = np.linspace(1, 20, 150)
    a_list = list(range(2, 70, 1))
    alpha_list = list(range(0, 181, 15))
    experiment([a], [alpha], SHOW=True, show_dynamic=True, opt=True, start_in_cntr=True,
               broadcasts_path="./my_weather1/10/"#, static_constraints_path="./my_weather1/10/"
               )#, save_to="./experiments_with_rotating/forVKR")
