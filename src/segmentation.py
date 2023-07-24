import cv2
import numpy as np

def segment_image(img: np.ndarray):
    """
    Get the colors from an image using KMeans clustering
    :param img: The image to get the colors from
    :return: (mask, colors) where mask  
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
    
    # return the mask and the colors
    return bestLabels.reshape(img.shape[:-1]).astype(np.uint8), centers
