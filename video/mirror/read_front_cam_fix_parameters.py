import csv


def read_front_cam_fix_parameters(path_parameters: str):
    """
    Read the parameters which are used to fix the mirror-image in the front-cam.

    :param path_parameters:
        Path to the file containing the fix-parameters. [mirror-center, mirror-radius, padding, warp top/bottom]
    :return:
        Dictionary containing the parameters per user. {user-id: fix-parameters}
    """

    parameters = {}

    with open(path_parameters, 'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')

        next(reader, None)

        for row in reader:
            [user_id, mc_x, mc_y, mr, padding, wt, wb] = row

            parameters[user_id] = {
                "mirror_center_x": int(mc_x) if mc_x is not '' else 0,
                "mirror_center_y": int(mc_y) if mc_y is not '' else 0,
                "mirror_radius": int(mr) if mr is not '' else 0,
                "padding": int(padding) if padding is not '' else 0,
                "warp_top": int(wt) if wt is not '' else 0,
                "warp_bottom": int(wb) if wb is not '' else 0
            }

    return parameters
