from skimage.measure import label, regionprops
import cv2
import image_processing
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


SHAPE_LITHO_DICT = {
    'circles': 'SAND/SANDSTONE',
    'rectangles': 'LIMESTONE',
    'ligns': 'CLAY',
    'vagues': 'MARL/MARLSTONE'
}
def filter_points_and_bounding_boxes(points, eps, min_samples):
    """
    Filters out points with less than two neighbors and computes bounding boxes for each cluster.

    Args:
    points (list): List of tuples containing the coordinates of the points (x, y).
    eps (float): The maximum distance between two points to be considered neighbors.
    min_samples (int): The minimum number of points required to form a dense region.

    Returns:
    list: List of filtered points.
    list: List of corresponding cluster labels.
    dict: Dictionary containing bounding boxes for each cluster.
    """
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3 to consider a cluster.")

    # Convert the list of tuples to a NumPy array
    points_np = np.array(points)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(points_np)

    # Filter out points that don't belong to any cluster (noise)
    filtered_points = points_np[dbscan.labels_ != -1]
    filtered_labels = dbscan.labels_[dbscan.labels_ != -1]

    # Compute bounding boxes for each cluster
    bounding_boxes = {}
    for label in np.unique(filtered_labels):
        cluster_points = filtered_points[filtered_labels == label]
        min_x, min_y = cluster_points.min(axis=0)
        max_x, max_y = cluster_points.max(axis=0)
        bounding_boxes[label] = {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}

    return filtered_points.tolist(), filtered_labels.tolist(), bounding_boxes


def filter_points(points, threshold_distance):
    # calculate pairwise distances between all points using a distance matrix
    distances = np.sqrt(((np.array(points)[:, None] - np.array(points)) ** 2).sum(axis=2))

    # create a mask for points that are close to other points
    mask = (distances <= threshold_distance).sum(axis=1) > 1

    # filter out the points that are far away from all other points
    return [p for i, p in enumerate(points) if mask[i]]


def bbox_center(bbox):
    """ Returns the center of a bounding box.

    Args:
        bbox (tuple): Tuple containing the coordinates of the bounding box (xmin, ymin, xmax, ymax).

    Returns:
        tuple: Tuple containing the coordinates of the center of the bounding box (x_center, y_center).
    """
    xmin, ymin, xmax, ymax = bbox
    x_center = round((xmin + xmax) / 2)
    y_center = round((ymin + ymax) / 2)
    return (y_center, x_center)


def filter_vertically_aligned_bboxes(bboxes, tolerance):
    """
    Filters bounding boxes to keep only those that are approximately aligned vertically.

    Args:
    bboxes (dict): Dictionary containing bounding boxes for each label.
    tolerance (int): Tolerance for the difference in average x-coordinates between bounding boxes.

    Returns:
    dict: Dictionary containing vertically aligned bounding boxes.
    """
    avg_x = [
        (label, (bbox['min_x'] + bbox['max_x']) / 2)
        for label, bbox in bboxes.items()
    ]
    avg_x.sort(key=lambda x: x[1])

    aligned_bboxes = {}
    for i, (label1, x1) in enumerate(avg_x[:-1]):
        for label2, x2 in avg_x[i + 1:]:
            if abs(x1 - x2) <= tolerance:
                aligned_bboxes[label1] = bboxes[label1]
                aligned_bboxes[label2] = bboxes[label2]
            else:
                break

    return aligned_bboxes


def contour_vague():
    # Load the image and convert it to grayscale
    image = cv2.imread('vague.png')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image to separate the foreground and background
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[20]

def task_detect_vague_circle(contour, gray,width, thresh, detect_circles, detect_vagues):

    is_circle = False
    circle_center = None

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    mask = cv2.bitwise_and(thresh, mask)
    black_pixels = np.count_nonzero(mask == 0)
    total_pixels = np.count_nonzero(mask)
    black_percentage = black_pixels / total_pixels

    # 3. Detect circles
    if detect_circles and black_percentage > 0.7:
        # Get the center and radius of the circle that would enclose the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # Calculate the area of the contour and the area of the circle that would enclose the contour
        contour_area = cv2.contourArea(contour)
        enclosing_area = np.pi * radius ** 2
        # Determine if the contour is a circle
        if abs(1 - (contour_area / enclosing_area)) < 0.3 and radius < 0.003 * width and radius > 0.0003 * width:
            # Do something with the contour, like draw a circle around it or fill it in
            circle_center = (int(x), int(y))
            is_circle = True
            #circles.append(circle_center)

    is_vague = bool(
        detect_vagues
        and cv2.matchShapes(
            contour_vague(), contour, cv2.CONTOURS_MATCH_I1, 0.0
        )
        < 0.3
    )
    return circle_center if is_circle else None, contour if is_vague else None

def detect(img_path: str, detect_vagues=False, detect_ligns=False, detect_rectangles=False, detect_circles=False,
           crop_height=None, display_image=True):

    # verify params
    if not detect_vagues and not detect_ligns and not detect_rectangles and not detect_circles:
        raise ValueError("At least one shape must be detected.")

    # Load the image and convert it to grayscale
    image = cv2.imread(img_path)
    print("Image base shape: ", image.shape)
    image = image_processing.denoise_image(image,"gaussian",coeff=1)
    if crop_height:
        image = image[crop_height[0]:crop_height[1], :]
    print("Image shape: ", image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the width of the image
    width = image.shape[1]
    result = {"base_column_height": image.shape[0]}
    # Create empty lists to store the detected shapes
    circles = []
    rectangles = []
    ligns = []
    vagues = []

    # Apply a threshold to the image to separate the foreground and background
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # 1. Detect rectangles
    if detect_rectangles:
        print("Detecting rectangles...")
        label_image = label(
            thresh)  # the label function returns a matrix with the same shape as the input image, where each pixel is labeled with a number corresponding to the connected component it belongs to (0 for the background, 1 for the first connected component, 2 for the second, etc.).
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.extent >= 0.85 and region.area < (width * 1.7) and region.area > (
                    width * 0.05) and region.axis_minor_length * 1.5 < region.axis_major_length and region.axis_major_length < width * 0.04:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                aspect_ratio = (maxc - minc) / (maxr - minr)
                # determine if the rectangle is horizontal or vertical
                if aspect_ratio > 1:
                    rectangles.append(region.bbox)

    # 2. Detect ligns
    if detect_ligns:
        print("Detecting ligns...")
        label_image = label(~thresh)
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if region.extent >= 0.75 and (maxr - minr) < 15 and (maxc - minc) > (width * 0.01) and (maxc - minc) < (
                    width * 0.04):
                ligns.append(region.bbox)

    print("Detecting vagues & circles...")
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Loop through each contour to determine if it's a circle and if it has more than 70% black pixels
    # use multithreading to speed up the process
    print("> Number of contours found: ", len(contours))
    n_workers = 3 if len(contours) < 4000 else 10
    with tqdm(total=len(contours)) as progress:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(task_detect_vague_circle, contour, gray, width, thresh, detect_circles, detect_vagues) for contour in contours]
            for future in as_completed(futures):
                progress.update() # We update the progress bar each time a task is completed
                circle_center, vague_contour = future.result()
                if circle_center is not None:
                    circles.append(circle_center)
                if vague_contour is not None:
                    vagues.append(vague_contour)

    """for contour in contours:
        # Calculate the percentage of black pixels in the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mask = cv2.bitwise_and(thresh, mask)
        black_pixels = np.count_nonzero(mask == 0)
        total_pixels = np.count_nonzero(mask)
        black_percentage = black_pixels / total_pixels

        # 3. Detect circles
        if detect_circles and black_percentage > 0.7:
            # Get the center and radius of the circle that would enclose the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # Calculate the area of the contour and the area of the circle that would enclose the contour
            contour_area = cv2.contourArea(contour)
            enclosing_area = np.pi * radius ** 2
            # Determine if the contour is a circle
            if abs(1 - (contour_area / enclosing_area)) < 0.3 and radius < 0.003 * width and radius > 0.0003 * width:
                # Do something with the contour, like draw a circle around it or fill it in
                center = (int(x), int(y))
                circles.append(center)

        # 4. Detect vagues
        if (
            detect_vagues
            and cv2.matchShapes(
                contour_vague(), contour, cv2.CONTOURS_MATCH_I1, 0.0
            )
            < 0.3
        ):
            vagues.append(contour)"""

    print("Filtering points...")
    # • Red Circles filter
    if detect_circles:
        circles = filter_points(circles, width * 0.02)  # filter out the points that are far away from all other points
        circles, circles_labels, circles_bbox = filter_points_and_bounding_boxes(circles, width * 0.02, 4)  # filters out points with less than two neighbors and computes bounding boxes for each cluster.
        circles_bbox = filter_vertically_aligned_bboxes(circles_bbox, width * 0.04)  # Filters out bounding boxes that are not vertically aligned
        result['circles'] = circles
        result['circles_labels'] = circles_labels
        result['circles_bbox'] = circles_bbox

    # • Green Rectangles filter
    if detect_rectangles:
        rectangles_centers = [bbox_center(rectangle) for rectangle in rectangles]
        rectangles_centers = filter_points(rectangles_centers, width * 0.06)
        _, rectangles_labels, rectangles_bbox = filter_points_and_bounding_boxes(rectangles_centers, width * 0.06, 3)
        rectangles_bbox = filter_vertically_aligned_bboxes(rectangles_bbox, width * 0.04) #todo
        result['rectangles'] = rectangles
        result['rectangles_labels'] = rectangles_labels
        result['rectangles_bbox'] = rectangles_bbox
        # d


    if display_image:
        print("Let's draw some shapes...")
        if detect_rectangles:
            for rectangle in rectangles:
                minr, minc, maxr, maxc = rectangle
                cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)  # vert
            for bbox in rectangles_bbox.values():
                cv2.rectangle(image, (bbox['min_x'], bbox['min_y']), (bbox['max_x'], bbox['max_y']), (0, 128, 0), 2)

            # stats
            print("> Number of green rectangles: ", len(rectangles))
            print("> Number of green rectangles clusters: ", len(rectangles_bbox))
        if detect_ligns:
            for lign in ligns:
                minr, minc, maxr, maxc = lign
                cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 255), 2)  # cyan

        if detect_circles:
            # Draw the circles
            for circle in circles:
                cv2.circle(image, circle, 2, (255, 0, 0), 2)
            # Draw the bounding boxes around the clusters
            for bbox in circles_bbox.values():
                cv2.rectangle(image, (bbox['min_x'], bbox['min_y']), (bbox['max_x'], bbox['max_y']), (255, 0, 0), 2)

            # stats
            print("> Number of red circles: ", len(circles))
            print("> Number of red circles clusters: ", len(circles_bbox))

        if detect_vagues:
            for vague in vagues:
                cv2.drawContours(image, vague, -1, (0, 0, 255), 2)

        display(Image.fromarray(image))

    return result,image.copy()
def height(bbox : dict):
    # bbox example => {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}
    return bbox['max_y'] - bbox['min_y']

def sum_height(bboxes : list):
    return sum(height(bbox) for bbox in bboxes)


def proportion(result : dict):

    base_height : int = result.get("base_column_height")
    keys = [k for k in result if 'bbox' in k]
    prop = {k: sum_height(result[k].values()) / base_height for k in keys}

    sum_prop = sum(prop.values())
    prop["others"] = 1 - sum_prop

    return prop


def proportion_litho(result : dict):
    """Change the name of the litho to the new name"""
    prop : dict = proportion(result)
    new_prop = {}

    for k,v in prop.items():
        k_cleaned = k.replace("_bbox","")
        litho = SHAPE_LITHO_DICT.get(k_cleaned,k)
        new_prop[litho] = v

    return new_prop

