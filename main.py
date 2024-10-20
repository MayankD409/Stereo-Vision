import argparse
from utils import Read_images, Feature_matching, Fundamental_and_Essential, compute_homographies, rectify_images, get_disparity, get_depth_map

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stereo Vision Processing Pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., classroom, storageroom, traproom)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    dataset_name = args.dataset

    print(f"Processing dataset: {dataset_name}")

    try:
        # Step 1: Read stereo images and camera parameters
        img1, img2, f, cx, cy, baseline_mm, width, height = Read_images(dataset_name)
        print("Images and calibration parameters loaded successfully.\n")
        
        # Step 2: Perform feature matching to find corresponding keypoints
        src_pts, dst_pts = Feature_matching(img1, img2)
        print("Feature matching completed.\n")
        
        # Step 3: Compute Fundamental matrix (F), Essential matrix (E), and camera matrix (K)
        F, E, K, R, t = Fundamental_and_Essential(src_pts, dst_pts, f, cx, cy)
        print("Fundamental and Essential matrices computed.\n")
        
        # Step 4: Get the height and width of the original images
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        
        # Step 5: Compute homography matrices for stereo rectification
        H1, H2 = compute_homographies(F, src_pts, dst_pts, w1, h1)
        print("Homography matrices computed.\n")
        
        # Step 6: Rectify the stereo images using computed homographies
        img1_rectified, img2_rectified = rectify_images(img1, img2, H1, H2, w1, h1, w2, h2, src_pts, dst_pts, F)
        print("Stereo rectification completed.\n")
        
        # Step 7: Compute disparity map from rectified stereo images
        disparity = get_disparity(img1_rectified, img2_rectified)
        print("Disparity map computed.\n")
        
        # Step 8: Generate and visualize the depth map from the disparity map
        depth_map = get_depth_map(disparity, f, baseline_mm)
        print("Depth map generated.\n")
        
        print("Stereo Vision Processing Pipeline completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
