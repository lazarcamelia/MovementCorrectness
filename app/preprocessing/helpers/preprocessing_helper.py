import math


def make_equal_length(timeseries, target_length):
    """
    Function used to equalize the number of frames for all the exercises
    :param timeseries: the current exercise as a timeseries data [nr_frames][nr_joints][nr_coordinates]
    :param target_length: the number of frames all the exercises to have
    """
    result_ts = timeseries[:]

    if len(timeseries) > target_length:
        # reduce the number of frames
        remove_count = len(result_ts) - target_length
        while remove_count > 0:
            step = int(len(result_ts) / remove_count)
            positions_to_remove = []
            for i in range(1, len(result_ts) - 1, step):
                positions_to_remove.append(i)

            while positions_to_remove:
                result_ts.pop(positions_to_remove.pop())
                if len(result_ts) == target_length:
                    break
            remove_count = len(result_ts) - target_length
    elif len(timeseries) < target_length:
        # add
        add_count = target_length - len(result_ts)
        while add_count > 0:
            add_count = min(add_count, len(result_ts))
            step = int(len(result_ts) / add_count)
            positions_to_add = []

            # find the positions where to add a new frame
            for i in range(1, len(result_ts) - 1, step):
                positions_to_add.append(i)

            while positions_to_add:
                result_ts = linear_interpolation(result_ts, positions_to_add.pop())
                if len(result_ts) == target_length:
                    break

            add_count = target_length - len(result_ts)
    return result_ts


def linear_interpolation(timeseries, current_position):
    """
    Function used to generate a new intermediary frame based on the neighbouring frames
    :param timeseries: the current exercise
    :param current_position: the position to insert a new frame
    :return: a new frame on the shape [nr_points][coordinates]
    """
    # new frame added between frame1 and frame2, at pos idx.
    frame1 = timeseries[current_position - 1]
    frame2 = timeseries[current_position]
    new_frame = []

    # parse each joint in the frames - 22 or 25 joints
    for joint in range(len(frame1)):
        new_pos = []
        for coord in range(3):  # Each landmark/point/joint has 3 coords: x, y, z.
            new_pos.append((frame1[joint][coord] + frame2[joint][coord]) / 2)
        new_frame.append(new_pos)

    timeseries.insert(current_position, new_frame)

    return timeseries


def z_score_normalization(scheleton_data):
    """
    Normalize skeleton data using z-score normalization (standardization).

    Args:
        scheleton_data (list): 4D list of shape (num_examples, num_timesteps, num_joints, 3)

    Returns:
        list: Normalized skeleton data
    """
    # Compute mean and std across all examples and timesteps
    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]
    total_points = 0

    for example in scheleton_data:
        for timestep in example:
            for joint in timestep:
                mean[0] += joint[0]
                mean[1] += joint[1]
                mean[2] += joint[2]
                total_points += 1

    mean[0] /= total_points
    mean[1] /= total_points
    mean[2] /= total_points

    for example in scheleton_data:
        for timestep in example:
            for joint in timestep:
                std[0] += (joint[0] - mean[0]) ** 2
                std[1] += (joint[1] - mean[1]) ** 2
                std[2] += (joint[2] - mean[2]) ** 2

    std[0] = math.sqrt(std[0] / total_points)
    std[1] = math.sqrt(std[1] / total_points)
    std[2] = math.sqrt(std[2] / total_points)

    # Normalize the data
    normalized_data = []
    for example in scheleton_data:
        normalized_example = []
        for timestep in example:
            normalized_timestep = []
            for joint in timestep:
                normalized_joint = [
                    (joint[0] - mean[0]) / (std[0] + 1e-8),
                    (joint[1] - mean[1]) / (std[1] + 1e-8),
                    (joint[2] - mean[2]) / (std[2] + 1e-8),
                ]
                normalized_timestep.append(normalized_joint)
            normalized_example.append(normalized_timestep)
        normalized_data.append(normalized_example)

    return normalized_data