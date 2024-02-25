import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def load_images_and_mask(source_path, target_path, mask_path):
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return source, target, binary_mask

def get_indices_and_boundary(binary_mask):
    omega_indices = np.array(np.where(binary_mask == 255)).T
    indices_dict = {tuple(idx): i for i, idx in enumerate(omega_indices)}
    num_pixels = len(omega_indices)
    N_p = {tuple(idx): [] for idx in omega_indices}
    boundary = set()
    height, width = binary_mask.shape
    for idx in omega_indices:
        i, j = idx
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < height and 0 <= y < width]
        N_p[tuple(idx)] = valid_neighbors
        boundary.update(n for n in valid_neighbors if n not in indices_dict)
    return omega_indices, indices_dict, num_pixels, N_p, boundary

def create_matrix_system(indices_dict, N_p, num_pixels):
    A = lil_matrix((num_pixels, num_pixels))
    for idx, i in indices_dict.items():
        A[i, i] = len(N_p[tuple(idx)])
        for n in N_p[tuple(idx)]:
            j = indices_dict.get(n)
            if j is not None:
                A[i, j] = -1
    return A.tocsr()

def calculate_v_pq(i, j, h_p, g_p, source, target, n, c):
    h_p_i = h_p[i].astype(np.int32)
    g_p_i = g_p[i].astype(np.int32)
    if j is not None:
        h_p_j = h_p[j].astype(np.int32)
        g_p_j = g_p[j].astype(np.int32)
        return np.where(np.abs(h_p_i - h_p_j) > np.abs(g_p_i - g_p_j), h_p_i - h_p_j, g_p_i - g_p_j)
    else:
        h_p_n = target[n][c].astype(np.int32)
        g_p_n = source[n][c].astype(np.int32)
        return np.where(np.abs(h_p_i - h_p_n) > np.abs(g_p_i - g_p_n), h_p_i - h_p_n, g_p_i - g_p_n)

def solve_and_replace(A, indices_dict, N_p, boundary, source, target, omega_indices):
    for c in range(3):
        h_p = target[omega_indices[:, 0], omega_indices[:, 1], c]
        g_p = source[omega_indices[:, 0], omega_indices[:, 1], c]
        b = np.zeros(A.shape[0])
        for idx, i in indices_dict.items():
            for n in N_p[idx]:
                j = indices_dict.get(n)
                v_pq = calculate_v_pq(i, j, h_p, g_p, source, target, n, c)
                b[i] += v_pq
                if n in boundary:
                    b[i] += target[n][c]
        f_p = spsolve(A, b)
        target[omega_indices[:, 0], omega_indices[:, 1], c] = np.clip(f_p, 0, 255).astype(np.uint8)
    return target

def main():
    source, target, binary_mask = load_images_and_mask('source.png', 'target.jpeg', 'mask.jpg')
    omega_indices, indices_dict, num_pixels, N_p, boundary = get_indices_and_boundary(binary_mask)
    A = create_matrix_system(indices_dict, N_p, num_pixels)
    result = solve_and_replace(A, indices_dict, N_p, boundary, source, target, omega_indices)
    cv2.imwrite('output.jpg', result)

if __name__ == "__main__":
    main()
