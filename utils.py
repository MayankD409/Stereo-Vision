import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Read images and camera parameters
def Read_images(dataset_name):
    img1_path = f'data/{dataset_name}/im0.png'
    img2_path = f'data/{dataset_name}/im1.png'
    calib_path = f'data/{dataset_name}/calib.txt'
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Images not found in dataset '{dataset_name}'. Please check the paths.")

    # Load calibration parameters from calib.txt
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        params = {}
        for line in lines:
            if '=' in line:
                key, value = line.strip().split('=')
                params[key.strip()] = value.strip()

    # Extract camera parameters
    cam0_str = params.get('cam0', '')
    cam1_str = params.get('cam1', '')
    if not cam0_str or not cam1_str:
        raise ValueError("Camera parameters 'cam0' or 'cam1' not found in calib.txt.")

    cam0_values = cam0_str.replace('[', '').replace(']', '').split(';')[0].strip().split()
    cam1_values = cam0_str.replace('[', '').replace(']', '').split(';')[1].strip().split()

    if len(cam0_values) < 3 or len(cam1_values) < 3:
        raise ValueError("Camera parameter values are incomplete in calib.txt.")

    f = float(cam0_values[0])  # Focal length in pixels
    cx = float(cam0_values[2])  # Principal point x-coordinate
    cy = float(cam1_values[2])  # Principal point y-coordinate

    baseline_mm = float(params.get('baseline', 0))  # Camera baseline in mm
    width = int(params.get('width', 0))  # Image width
    height = int(params.get('height', 0))  # Image height

    if baseline_mm == 0 or width == 0 or height == 0:
        raise ValueError("Baseline, width, or height parameters are missing or invalid in calib.txt.")

    return img1, img2, f, cx, cy, baseline_mm, width, height

# Perform feature matching between two images
def Feature_matching(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    if descriptors_1 is None or descriptors_2 is None:
        raise ValueError("No descriptors found in one or both images.")

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"Total matches found: {len(matches)}")
    print(f"Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < 10:
        raise ValueError("Not enough good matches found. Try capturing more images or different viewpoints.")

    # Extract location of good matches
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts

# Compute Fundamental and Essential matrices
def Fundamental_and_Essential(src_pts, dst_pts, f, cx, cy):
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 0.1, 0.99)
    
    if F is None:
        raise ValueError("Fundamental matrix computation failed.")
    
    # Construct camera matrix
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    E = K.T @ F @ K  # Compute Essential matrix

    # Decompose Essential matrix to get R and t (rotation and translation)
    points, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K)
    
    print("Fundamental Matrix F:")
    print(F)
    print("\nEssential Matrix E:")
    print(E)
    print("\nRotational Matrix R:")
    print(R)
    print("\nTranslation Vector t:")
    print(t)
    
    return F, E, K, R, t

# Compute homography matrices for stereo rectification
def compute_homographies(F, src_pts, dst_pts, w1, h1):
    # Perform stereo rectification
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(src_pts), np.float32(dst_pts), F, imgSize=(w1, h1))
    
    if not retval:
        raise ValueError("Stereo rectification failed.")

    print("Homography Matrix H1:")
    print(H1)
    print("\nHomography Matrix H2:")
    print(H2)
    
    return H1, H2

# Draw epipolar lines on images
def draw_epipolar(img1, img2, src, dst, w, F):
    img1_g = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw keypoints
    for pt in src:
        pt_int = (int(pt[0][0]), int(pt[0][1]))  # Convert to integer format
        cv2.circle(img1_g, pt_int, 5, (0, 255, 0), -1)
    for pt in dst:
        pt_int = (int(pt[0][0]), int(pt[0][1]))  # Convert to integer format
        cv2.circle(img2_g, pt_int, 5, (0, 255, 0), -1)

    # Compute epilines in img1 for points in img2
    lines1 = cv2.computeCorrespondEpilines(dst.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    for r, pt1, pt2 in zip(lines1, src, dst):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
        img1_g = cv2.line(img1_g, (x0, y0), (x1, y1), color, 1)
        img1_g = cv2.circle(img1_g, (int(pt1[0][0]), int(pt1[0][1])), 5, color, -1)
        img2_g = cv2.line(img2_g, (x0, y0), (x1, y1), color, 1)
        img2_g = cv2.circle(img2_g, (int(pt2[0][0]), int(pt2[0][1])), 5, color, -1)

    return img1_g, img2_g

# Rectify the stereo images using the computed homographies
def rectify_images(img1, img2, H1, H2, w1, h1, w2, h2, src_pts, dst_pts, F):
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    
    # Draw epipolar lines on rectified images
    i1, i2 = draw_epipolar(img1_rectified, img2_rectified, src_pts, dst_pts, w1, F)
    
    concatenated_img = np.concatenate((i1, i2), axis=1)
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB))
    plt.title('Rectified Images with Epipolar Lines')
    plt.axis('off')
    plt.show()
    
    return img1_rectified, img2_rectified

# Compute disparity map from rectified stereo images
def get_disparity(img1_rectified, img2_rectified):
    stereo = cv2.StereoBM_create(numDisparities=16*6, blockSize=15)  # Increased numDisparities for better results

    # Compute the disparity map
    disparity = stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0

    # Normalize the disparity map for visualization
    disparity_rescaled = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_rescaled = np.uint8(disparity_rescaled)

    # Show the rescaled disparity map as grayscale
    plt.figure(figsize=(10, 5))
    plt.title('Disparity Map (Grayscale)')
    plt.imshow(disparity_rescaled, 'gray')
    plt.axis('off')
    plt.show()

    # Create a color map for visualization
    disparity_color = cv2.applyColorMap(disparity_rescaled, cv2.COLORMAP_JET)

    # Show the colorized disparity map
    plt.figure(figsize=(10, 5))
    plt.title('Disparity Map (Color)')
    plt.imshow(cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return disparity

# Generate and visualize the depth map from the disparity map
def get_depth_map(disparity, f, baseline_mm):
    # Avoid division by zero and invalid disparity values
    with np.errstate(divide='ignore'):
        depth_map = (f * baseline_mm) / disparity
    depth_map[disparity <= 0] = 0  # Set depth to zero where disparity is invalid

    # Define maximum depth for visualization
    max_depth = 1000  # in millimeters
    depth_map_clipped = np.clip(depth_map, 0, max_depth)

    # Normalize the depth map for visualization
    depth_map_rescaled = cv2.normalize(depth_map_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth_map_rescaled = depth_map_rescaled.astype(np.uint8)

    # Create a colorized depth map for visualization
    depth_map_color = cv2.applyColorMap(depth_map_rescaled, cv2.COLORMAP_JET)

    # Show the depth map as grayscale
    plt.figure(figsize=(10, 5))
    plt.title('Depth Map (Grayscale)')
    plt.imshow(depth_map_rescaled, 'gray')
    plt.axis('off')
    plt.show()

    # Show the colorized depth map
    plt.figure(figsize=(10, 5))
    plt.title('Depth Map (Color)')
    plt.imshow(cv2.cvtColor(depth_map_color, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return depth_map

