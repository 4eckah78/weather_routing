from main import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time


def experiment(a_list, alpha_list, SHOW=False, opt=False, broadcasts_path="txt/"):
    result = np.zeros((len(a_list), len(alpha_list)))
    for i, a in enumerate(a_list):
        print(f"a = {a}")
        for j, alpha in enumerate(alpha_list):
            # print(f"alpha={alpha}")
            begin = time.time()
            broadcast = 1
            if broadcasts_path != "txt/":
                polygons = read_polygons_from(broadcasts_path + f"broadcast{broadcast}.txt")
                width, height = get_image_size_by_polygons(polygons, a)
                broadcast += 1
            else:
                polygons = read_polygons_from(polygons_file)

            polygons = scale_polygons_coords_up_left_corner(polygons, scale_up_left_corner, scale_up_left_corner)

            width, height = get_image_size_by_polygons(polygons, a)
            if USE_FIXED_SIZES:
                width, height = 640, 480

            start = (scale_up_left_corner, scale_up_left_corner)
            end = (width - 1 + scale_up_left_corner, height - 1 + scale_up_left_corner)

            # turn to 30 degree (optimal)
            if opt:
                betta = np.rad2deg(np.arctan(-end[1] / end[0]))
                opt_alpha = -30 - betta
                alpha = opt_alpha

            # print(f"distance = {np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)}")
            start_end = [start[0], start[1], end[0], end[1]]

            # center = [height // 2, width // 2]
            # rotated_polygons, start, end = rotate_polygons_around_center(polygons, start_end, center, alpha)
            # draw_start_pixel = tuple(start)
            # draw_end_pixel = tuple(end)
            # new_width, new_height = get_image_size_by_polygons(rotated_polygons, a)
            # rotated_polygons = rotated_polygons[:-1]
            # if SHOW_POLYGONS:
            #     show_polygons(polygons, width, height)
            #     show_polygons(rotated_polygons, new_width, new_height)
            #
            # polygons = rotated_polygons
            width, height = 1000, 1000
            # print(f"height={height}, width={width}")

            hex_width = math.ceil((width + scale_down_right_corner) / (3 * a))
            hex_height = math.ceil((height + scale_down_right_corner) / (np.sqrt(3) * a) * 2) + 1
            print(f"hex width x height: {hex_width}x{hex_height}")
            print(f"pixel width x height: {width}x{height}")
            static_hex_map = np.zeros((hex_height, hex_width), dtype=np.int32)

            start = pixel_to_flat_hex(start[0], start[1], a)
            end = pixel_to_flat_hex(end[0], end[1], a)
            hex_polygons, static_hex_map = from_pixel_to_hex_polygons(polygons, static_hex_map, a)
            static_hex_map = raster_hex_polygons(hex_polygons, static_hex_map, polygons, width, height, hex_width, hex_height, a)
            draw_hex_image([start, end], "red", width, height, hex_width, hex_height, static_hex_map, a,
                           polygons=polygons)

            start = pixel_to_flat_hex(start[0], start[1], a)
            end = pixel_to_flat_hex(end[0], end[1], a)
            visited = [{start}]

            colors = ["yellow", "purple", "green", "pink", "gray", "blue", "lime", "orange", "salmon",
                      "silver", "teal", "violet", "wheat"]

            max_iterations = static_hex_map.shape[0] * static_hex_map.shape[1]
            iterations = 0
            finished = False
            dynamic_hex_map = np.copy(static_hex_map)
            colour = 0
            draw_hex_image([start, end], "red", width, height, hex_width, hex_height, dynamic_hex_map, a, polygons=polygons)
            while not finished and iterations < max_iterations:
                dynamic_hex_map = np.copy(static_hex_map)
                dynamic_hex_map = update_map(f"{broadcasts_path}broadcast{broadcast}.txt", dynamic_hex_map, a, alpha=alpha)
                broadcast += 1
                draw_hex_image([start, end], "red", width, height, hex_width, hex_height, dynamic_hex_map, a)
                bg = time.time()
                finished = move(visited, dynamic_hex_map, end)
                if finished:
                    paths = move_back(visited, end, dynamic_hex_map)
                    to_color = set.union(*paths)
                    # path_to_ship = get_paths(paths, dynamic_hex_map)
                    if SHOW:
                        broadcast = 1
                        to_color = set([start])
                        paths.pop()
                        paths = paths[::-1]
                        t = math.ceil(len(paths) / distance)
                        # draw_hex_image({(0,0)}, "red", width, height, hex_width, hex_height, dynamic_hex_map, save_to=f"demo_dyn_2/broadcast{0}.png")
                        for step in range(0, distance * t, distance):
                            layer = paths[step:step + distance]
                            if broadcast < 6:
                                dynamic_hex_map = np.copy(static_hex_map)
                                dynamic_hex_map = update_map(f"{broadcasts_path}broadcast{broadcast}.txt", dynamic_hex_map, a,
                                                             alpha=alpha)
                                broadcast += 1
                            draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                                           dynamic_hex_map,
                                           a)
                            new_layer = set.union(*layer)
                            to_color = set.union(to_color, new_layer)
                            draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                                           dynamic_hex_map,
                                           a)
                        # draw_hex_image(to_color, "red", width, height, hex_width, hex_height,
                        #                dynamic_hex_map,
                        #                a)#, draw_start_end_pixels=[draw_start_pixel,
                        #                        draw_end_pixel]),  # save_to=f"experiments_with_rotating/experiment_{alpha}_degree_{a}_hex.png")
                    # print(f"optimal path's length in hexes = {len(paths) - 1}")
                    stats_hex.append(len(paths) - 1)
                    path_length_in_pixels = round((len(paths) - 1) * np.sqrt(3) * a)
                    stats.append(path_length_in_pixels)
                    # print(f"optimal path's length in pixels = {path_length_in_pixels}")
                    result[i][j] = path_length_in_pixels
                    # print(f"alpha={alpha}")
                else:
                    to_color = set.union(*visited[-distance:])
                    # draw_hex_image(to_color, colors[colour % len(colors)], width, height, hex_width, hex_height, dynamic_hex_map)
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
    a_list = list(range(2, 21, 1))
    alpha_list = list(range(0, 360, 3))
    # creating real broadcasts
    # for i in range(1, 21):
    #     create_broadcasts_from_folder(f"./weather1/{i}", save_path=f"./my_weather1/{i}")
    # for i in range(1, 21):
    #     create_broadcasts_from_folder(f"./weather1/{i}", save_path=f"./my_weather2/{i}")
    # create_broadcasts_from_folder("./weather1/9", save_path=f"./test")
    experiment([a], [alpha], SHOW=True, opt=True, broadcasts_path="./my_weather1/9/")


    # experiment(a_list, [alpha])
    # plot_graphic(a_list, stats, f"l(a) with dynamic constraints (alpha={alpha})", "a", "length in pixels")
    # stats = []
    # experiment([a], alpha_list)
    # plot_graphic(alpha_list, stats, f"l(alpha) with dynamic constraints (a={a})", "alpha", "length in pixels")
    # a_list = list(range(10, 50, 3))
    # alpha_list = list(range(0, 360, 10))
    # plot_3D(a_list, alpha_list, experiment)

    # with open("time_stats.txt", "w") as f:
    #     f.write(",".join([str(t) for t in time_stats]))
    # with open("time_stats.txt", "r") as f:
    #     time_stats = list(map(float, f.readline().split(',')))
    # for time complexity
    # with open("time_stats.txt", "r") as f:
    #     time_stats = list(map(float, f.readline().split(',')))
    # with open("time_stats.txt", "w") as f:
    #     for alpha in [0, 10, 20, 30, 40, 50]:
    #         time_stats = []
    #         experiment(a_list, [alpha], opt=True)
    #         f.write(",".join([str(t) for t in time_stats]))
    #         f.write("\n")
    #         plt.plot(a_list, time_stats, label=f"alpha={alpha}")

    # for time complexity
    # from scipy.optimize import curve_fit
    #
    #
    # def func(t, k, a_, x0):
    #     return k * np.exp(-a_ * (t - x0))

    # a_list = np.array(a_list)
    # time_stats = np.array(time_stats[:len(a_list)])

    # popt, pcov = curve_fit(func, a_list, time_stats, (1., 1., 1.), maxfev=10 ** 6)
    # k, a_param, x0 = popt
    #
    # print('k={0}\na={1}\nx0={2}\n'.format(*tuple(popt)))

    # plt.plot(a_list, time_stats, 'ro', markersize=3, label=f"empiric")
    # plt.plot(np.linspace(2, 20, 100), 339.1 * np.exp(-1.154 * np.linspace(2, 20, 100)), 'b', label=f"theoretic")
    # plt.legend()
    # plt.xlabel("a"), plt.ylabel("time in seconds")
    # # plt.title(f"time complexity, {round(k, 2)}*e^(-{round(a_param, 2)}*(x + {-1*round(x0, 2)}))")
    # plt.title(f"time complexity, {339.1}*e^(-{1.154}x)")
    # plt.show()
    # print(f"koeff={k*np.exp(a_param*x0)}")

    # plot_3D(a_list, alpha_list, experiment)
    # experiment(a_list, [alpha])
    # with open("stats.txt", "r") as f:
    #     stats = list(map(int, f.readline().split(',')))
    # with open("stats.txt", "w") as f:
    #     f.write(",".join([str(length) for length in stats]))
    #     f.write("\n")
    #     f.write(",".join([str(a) for a in a_list]))
    # zzip = zip(stats, a_list)
    # zzip = sorted(zzip, key=lambda x: x[0])
    # plot_graphic(a_list, stats, f"Pixel_length(a), alpha={alpha}", "a", "pixel length")

    # for min_path, a in zzip[:3]:
    #     print(f"min path = {min_path} pixels, a = {a}")
    # experiment([a], [alpha], SHOW=True)
    # for max_path, a in zzip[-3:]:
    #     print(f"max path = {max_path} pixels, a = {a}")
    #     #experiment([a], [alpha], SHOW=True)

    # min_el = zzip[0][0]
    # min_alphas = [alpha for length, alpha in zzip if length == min_el]
    # print(f"min_alphas={min_alphas}")
    # print(min_el)
    # print((max(stats) - min(stats)) // 2)
    # # for alpha in min_alphas:
    # #     experiment([a], [alpha])
    # import math

    # rad = [math.radians(grad) for grad in alpha_list]
    # b = [np.arcsin((y-866)/69) - np.pi/3*x for x, y in zip(rad, stats)]
    # plt.plot(rad, b, 'b')
    # plt.plot(rad, [np.mean(b) for _ in range(len(b))])
    # print(np.mean(b))
    # print(np.rad2deg(np.mean(b)))
    # plt.show()
    # sin = [69*np.sin(np.pi*x/3*2*np.pi - (5*np.pi/6 + np.pi/10)) + 866 for x in rad]
    # plt.plot(a_list, stats, label="l(alpha)"), plt.title(f"l = length_of_optimal_route_in_pixels(alpha) (a = {a})")
    # plt.plot(a_list, sin, label="teor"), plt.title(f"l = length_of_optimal_route_in_pixels(alpha) (a = {a})")
    # plt.plot(rad, [np.mean(stats) for _ in range(len(stats))], label="mean")
    # plt.legend()
    # plt.xlabel("alpha"), plt.ylabel("length in pixels")
    # plt.grid()
    # plt.show()

    # k = [490 / a for a in alpha_list]
    # plt.plot(alpha_list, k, 'b', label="theoretical hex length")
    # plt.plot(a_list, stats_hex)#, label="empiric hex length")
    # plt.title(f"l = length_of_optimal_route_in_hexes(alpha) (a = {a})")
    # plt.xlabel("alpha"), plt.ylabel("length in hexes")
    # plt.legend()
    # plt.grid()
    # plt.show()
