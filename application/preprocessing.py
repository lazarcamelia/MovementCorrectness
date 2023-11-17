import numpy as np
import torch
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
import torch
import os

def extract_info_from_filename(filename):
  '''
    split the filename and return the relevant info
  '''
  pieces = filename.split("_")

  movement_id = pieces[0]
  subject_id = pieces[1]
  exercise_id = pieces[2]

  label = 1

  if len(pieces) == 5:
    label = 0

  return movement_id, subject_id, exercise_id, label


# read a txt file
def read_df(filename):
  df = pd.read_csv(filename, sep = ',', header=None, index_col=False)
  tensor_df = torch.tensor(df.values)

  tensor_size = tensor_df.size(0)

  # resize the tensor from [66] to [22][3]
  resized_tensor = tensor_df.view(tensor_size, 22, 3)

  return resized_tensor


DESIRED_NR_POINTS = 75
NR_JOINTS = 22

def preprocess_tensor(tensor):
  '''
    Function used for preprocessing the tensor. The function applied are:
    - if the tensor size (nr frames) < DESIRED_NR_POINTS - we apply upsampling using cubic interpolation
    - if the tensor size (nr frames) > DESIRED_NR_POINTS - we apply downsapling by removing some of the points
  '''
  tensor_size = tensor.size(0)

  if tensor_size < DESIRED_NR_POINTS:
    return cubic_interpolation_for_all_joints(tensor)
  elif tensor_size > DESIRED_NR_POINTS:
    return downsampling(tensor)
  else:
    return tensor


def cubic_interpolation_for_all_joints(tensor):
  '''
    Function used for computing new coordinates for the joints on intermediary frames
    We compute the new joint coordinates for each joint separately, as their position is indepenent
  '''

  interpolated_tensors = []

  for i in range(NR_JOINTS):
    extracted_joint = tensor[:, i, :]
    # interpolated_joint = cubic_interpolation(extracted_joint)
    interpolated_joint = cubic_interpolation_with_addition_only(extracted_joint)
    interpolated_tensors.append(interpolated_joint)

  resulted_tensor = torch.stack(interpolated_tensors, axis=1)

  return resulted_tensor


def cubic_interpolation(tensor_joint):
  '''
    Cubic interpolation applied at one joint at the time.
    We apply 3D interpolation on all 3 coordinates.
    This implementation gets rid of all the original points
  '''

  # Calculate cubic spline functions for each dimension (x, y, z)
  x_values = tensor_joint[:, 0]
  x_cspline = CubicSpline(np.arange(len(x_values)), x_values)

  y_values = tensor_joint[:, 1]
  y_cspline = CubicSpline(np.arange(len(y_values)), y_values)

  z_values = tensor_joint[:, 2]
  z_cspline = CubicSpline(np.arange(len(z_values)), z_values)


  # Calculate the interval between new points
  interval = len(x_values) / (DESIRED_NR_POINTS - 1)

  # Initialize upsampled points array
  upsampled_points = []

  # Interpolate and generate new points
  for i in range(DESIRED_NR_POINTS):
      index = i * interval
      x_interpolated = x_cspline(index)
      y_interpolated = y_cspline(index)
      z_interpolated = z_cspline(index)
      upsampled_points.append([x_interpolated, y_interpolated, z_interpolated])

  upsampled_points = np.array(upsampled_points)
  # Convert the upsampled points list to a numpy array
  upsampled_points_tensor = torch.tensor(upsampled_points)

  return upsampled_points_tensor


def cubic_interpolation_with_addition_only(tensor_joint):
  '''
    Cubic interpolation applied at one joint at the time.
    We apply 3D interpolation on all 3 coordinates.
  '''

  tensor_initial_size = tensor_joint.size(0)

  # Calculate cubic spline functions for each dimension (x, y, z)
  x_values = tensor_joint[:, 0]
  x_cspline = CubicSpline(np.arange(len(x_values)), x_values)

  y_values = tensor_joint[:, 1]
  y_cspline = CubicSpline(np.arange(len(y_values)), y_values)

  z_values = tensor_joint[:, 2]
  z_cspline = CubicSpline(np.arange(len(z_values)), z_values)


  # Calculate the interval between original points
  interval = (tensor_initial_size - 1) / (DESIRED_NR_POINTS - tensor_initial_size)

  upsampled_points = tensor_joint.clone()

  # for i in range(len(tensor_joint)):
    # upsampled_points.append(tensor_joint[i])
    # nr_added_initially += 1
    # if i < len(tensor_joint) - 1:
    #     for j in range(1, int(interval)):
    #         t = i + j * interval
    #         interpolated_point = torch.tensor([
    #             x_cspline(t),
    #             y_cspline(t),
    #             z_cspline(t)
    #         ])
    #         upsampled_points.append(interpolated_point)
    #         new_generated += 1

  for i in range(tensor_initial_size, DESIRED_NR_POINTS):
    t = tensor_initial_size + (i - tensor_initial_size) * interval

    x_interpolated = x_cspline(t)
    y_interpolated = y_cspline(t)
    z_interpolated = z_cspline(t)

    # print("X is : " + str(x_interpolated))
    # print("Y is : " + str(y_interpolated))
    # print("Z is : " + str(z_interpolated))

    interpolated_point = torch.tensor(np.array([
        x_interpolated,
        y_interpolated,
        z_interpolated,
    ]))

    print("The pos generated is " + str(t))

    upsampled_points = torch.cat((upsampled_points[:i], interpolated_point.view(1, 3), upsampled_points[i:]))

  # upsampled_points = np.array(upsampled_points)
  # Convert the upsampled points list to a numpy array
  # upsampled_points_tensor = torch.tensor(upsampled_points)
  # upsampled_points_tensor = torch.stack(upsampled_points)
  return upsampled_points


def downsampling(tensor):
  '''
    Function used for downsampling the frames. We select positions from the tensor from where we will remove the frames.
    The positions selected are uniformly distributed among the frames
  '''
  # Calculate the stride for downsampling
  stride = tensor.shape[0] // DESIRED_NR_POINTS
  elements_between_cuts = int(tensor.shape[0] // (DESIRED_NR_POINTS - 1))
  indices_to_keep = []

  for i in range(DESIRED_NR_POINTS):
    index = int(i * elements_between_cuts)
    indices_to_keep.append(index)

  downsampled_tensor = tensor[indices_to_keep]

  return downsampled_tensor

def read_data():
    BASE_PATH = "/content/drive/My Drive/Masters/Disertatie/Datasets/UI-PRMD/"

    all_files = os.listdir(BASE_PATH)
    all_files_size = len(all_files)
    all_exercises = torch.zeros(all_files_size, DESIRED_NR_POINTS, 22, 3)
    labels = torch.zeros(all_files_size, dtype=torch.float32)

    # count the number of elements in each tensor to get some stats
    tensor_sizes =  torch.zeros(all_files_size, dtype=torch.long)

    index_exercise = 0

    for filename in all_files:
      if 'txt' not in filename:
        continue

      movement_id, subject_id, exercise_id, label = extract_info_from_filename(filename)
      tensor = read_df(BASE_PATH + filename)
      tensor_size = tensor.size(0)

      print("Size of initial data is " + str(tensor.size()))
      processed_tensor = preprocess_tensor(tensor)
      print("Size of reshaped data is " + str(tensor.size()))

      all_exercises[index_exercise] = processed_tensor

      tensor_sizes[index_exercise] = tensor_size
      labels[index_exercise] = label

      index_exercise += 1