import numpy as np

default_vertices = np.array([
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [1,  1, -1, -1,  1, 1, -1, -1],
    [1,  1,  1,  1, -1, -1, -1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1]
])

scale_matrix_A = np.array([
    [0.5, 0, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0, 0.5, 0],
    [0, 0, 0, 1]
])

scale_matrix_B = np.array([
    [2, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 1]
])

scale_matrix_C = np.array([
    [1.5, 0, 0, 0],
    [0, 1.5, 0, 0],
    [0, 0, 0.7, 0],
    [0, 0, 0, 1]
])

translation_matrix_A = np.array([
    [1, 0, 0, 3],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

translation_matrix_B = np.array([
    [1, 0, 0, -2.5],
    [0, 1, 0, -3.5],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

translation_matrix_C = np.array([
    [1, 0, 0, -2],
    [0, 1, 0, 2.5],
    [0, 0, 1, 2.5],
    [0, 0, 0, 1]
])

translation_matrix_D = np.array([
    [1, 0, 0, -3],
    [0, 1, 0, 2.5],
    [0, 0, 1, -4],
    [0, 0, 0, 1]
])

translation_matrix_E = np.array([
    [1, 0, 0, 5],
    [0, 1, 0, -0.5],
    [0, 0, 1, -1.5],
    [0, 0, 0, 1]
])

rotate_matrix_X_axis = np.array([
    [1, 0, 0, 0],
    [0, np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
    [0, np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
    [0, 0, 0, 1]
])

rotate_matrix_Y_axis = np.array([
    [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
    [0, 1, 0, 0],
    [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4), 0],
    [0, 0, 0, 1]
])

rotate_matrix_Z_axis = np.array([
    [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0, 0],
    [np.sin(np.pi / 4), np.cos(np.pi / 4), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

translation_matrix_v_1_1_1 = np.array([
    [1, 0, 0, -1],
    [0, 1, 0, -1],
    [0, 0, 1, -1],
    [0, 0, 0, 1]
])

inversed_translation_matrix_v_1_1_1 = np.array([
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 1]
])