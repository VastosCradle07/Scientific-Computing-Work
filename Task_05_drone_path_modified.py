import numpy as np

path = np.array([
    [0.0, 0.0, 1.0],
    [2.0, 0.0, 1.0],
    [2.0, 1.0, 1.0]
])

angle = np.deg2rad(30)

rotation = np.array([
    [np.cos(angle), -np.sin(angle), 0.0],
    [np.sin(angle), np.cos(angle), 0.0],
    [0.0, 0.0, 1.0]
])

scale = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 1.0]
])

translation = np.array([
    [1.0, 0.0, 4.0],
    [0.0, 1.0, 3.0],
    [0.0, 0.0, 1.0]
])

transformation = translation @ rotation @ scale

transformed_path = path @ transformation.T

recovered_path = transformed_path @ np.linalg.inv(transformation).T

print("Original waypoints:\n", path[:, :2])
print("Transformation matrix:\n", np.round(transformation, 4))
print("Transformed waypoints:\n", np.round(transformed_path[:, :2], 4))
print("Determinant:", np.linalg.det(transformation))
print("Inverse recovers original path:", np.allclose(recovered_path, path))
