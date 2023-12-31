import cv2
import numpy as np

def segment_image(img: np.ndarray):
    """
    Get the colors from an image using KMeans clustering
    :param img: The image to get the colors from. Should be a 3D array of shape (height, width, 3). The color values should be in range [0,255], and the image can be either BGR or RGB. The image should already have the background masked out before being passed to this function. Failing to do this won't cause an error but may reduce the robustness or accuracy of the results of this function.
    :return: (mask, colors) where mask is a 2D array of the same shape as img, and colors is a 3x3 array of the colors
    mask is 0 for the background, 1 where the shape is, and 2 where the letter is
    colors[0] is the color of the background, colors[1] is the color of the shape, colors[2] is the color of the letter 
    """
    # Reshape the image to be a list of pixels
    pixel_list = img.reshape((img.shape[0] * img.shape[1], 3))
    coords_list = np.indices((img.shape[0], img.shape[1])).reshape(2, -1).T
    pixel_coords_list = coords_list - np.array([img.shape[0] / 2, img.shape[1] / 2])
    dist_from_center = np.linalg.norm(pixel_coords_list, axis=1)
    augmented_data = np.concatenate((pixel_list, dist_from_center[:, np.newaxis]), axis=1)

    # n_clusters hard-coded at 3 
    retval, bestLabels, centers = cv2.kmeans(
        data = augmented_data.astype(np.float32), 
        K = 3, 
        bestLabels = None, 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 
        attempts = 10, 
        flags = cv2.KMEANS_RANDOM_CENTERS)
    
    # sort the centers by distance from the center of the image
    sorted_indices = np.flip(np.argsort(centers[:, 3]))

    new_labels = np.zeros_like(bestLabels)
    # re-assign labels so the furthest cluster is 0, the closest is 2
    for new_label, old_label in enumerate(sorted_indices):
        new_labels[bestLabels == old_label] = new_label

    centers = centers[sorted_indices, :3]

    # return the mask and the colors
    return new_labels.reshape(img.shape[:-1]).astype(np.uint8), centers.astype(np.uint8)
