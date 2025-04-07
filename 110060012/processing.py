import os
import numpy as np
from rembg import remove
from PIL import Image
from io import BytesIO
import cv2
from sklearn.preprocessing import normalize

def remove_background(input_path, remove_background_path, background_path):
    """
    Removes the background of the input image using rembg, saves the background-removed image and 
    the background image separately. If they already exist, loads them directly.
    
    Parameters:
    - input_path (str): Path to the input image.
    - remove_background_path (str): Path to save the background-removed image.
    - background_path (str): Path to save the background image.
    
    Returns:
    - PIL.Image: The background-removed image as a PIL Image object.
    """
    # Check if the background-removed image already exists
    if os.path.exists(remove_background_path) and os.path.exists(background_path):
        bg_removed_image = Image.open(remove_background_path)
        # print(f"Background removed image loaded from {remove_background_path}")
        return bg_removed_image
    
    # Open the input image
    input_image = Image.open(input_path).convert("RGB")  # Ensure it's RGB
    input_image_np = np.array(input_image)

    # Convert the image to bytes to pass it to rembg
    img_byte_arr = BytesIO()
    input_image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    
    # Remove the background using rembg
    bg_removed = remove(img_byte_arr)

    # Convert the bytes back to an image
    bg_removed_image = Image.open(BytesIO(bg_removed)).convert("RGBA")  # Ensure RGBA for transparency handling
    bg_removed_image.save(remove_background_path, format="PNG")
    print(f"Background removed image saved to {remove_background_path}")

    # Convert the background-removed image to RGB for subtraction
    bg_removed_image_np = np.array(bg_removed_image.convert("RGB"))  # Strip alpha channel

    # Handle grayscale images
    if len(input_image_np.shape) == 2:  # Grayscale input
        input_image_np = np.stack([input_image_np] * 3, axis=-1)  # Convert to RGB-like shape
    
    # Subtract the background
    bg_image_np = np.clip(input_image_np - bg_removed_image_np, 0, 255).astype(np.uint8)

    # Convert the result back to an image and save it
    bg_image = Image.fromarray(bg_image_np)
    bg_image.save(background_path)
    print(f"Background image saved to {background_path}")
    
    return bg_removed_image

def decolor_image(input_path, output_path, remove_background_path, background_path):
    """
    Decolorizes an image after removing its background and saves it.
    """
    if os.path.exists(output_path):
        print(f"Decolorized image already exists at {output_path}")
        return np.array(Image.open(output_path))
    
    # Remove the background first
    bg_removed_image = remove_background(input_path, remove_background_path, background_path)
    
    # Convert to grayscale
    grayscale_image = bg_removed_image.convert("L")
    
    # Save the grayscale image
    grayscale_image.save(output_path)
    print(f"Decolorized image saved to {output_path}")
    
    return np.array(grayscale_image)


def resize_image(image, target_size):
    """
    Resize the image to match the target size using OpenCV.
    """
    return cv2.resize(image, target_size)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def recolor_image(target_path, reference_paths, output_path):
    """
    Recolors the grayscale target image based on the histogram of multiple reference RGB images.

    Parameters:
    - target_path (str): Path to the grayscale target image.
    - reference_paths (list[str]): List of paths to reference RGB images.
    - output_path (str): Path to save the recolored RGB image.
    """
    # Load the target grayscale image
    
    if os.path.exists(output_path):
        print(f"Recolored image already exists at {output_path}")
        return
    
    target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if target_image is None:
        raise ValueError(f"Could not load target image from {target_path}")
    print(f"Target Image Shape: {target_image.shape}")

    # Ensure dimensions are correct
    height, width = target_image.shape

    # Load and preprocess the reference images
    reference_images = []
    for ref_path in reference_paths:
        ref_image = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref_image is None:
            print(f"Skipping invalid reference image at {ref_path}")
            continue
        # Resize reference images to match the target dimensions
        ref_image = cv2.resize(ref_image, (width, height))
        reference_images.append(ref_image)

    if not reference_images:
        raise ValueError("No valid reference images provided.")
    print(f"Number of valid reference images: {len(reference_images)}")

    # Calculate average histograms for R, G, and B channels across reference images
    combined_histograms = []
    for i in range(3):  # RGB channels
        channel_histogram = np.zeros(256)
        for ref_image in reference_images:
            ref_hist = cv2.calcHist([ref_image], [i], None, [256], [0, 256])
            channel_histogram += ref_hist[:, 0]
        combined_histograms.append(channel_histogram / len(reference_images))
        print(f"Combined Histogram for Channel {i}: {combined_histograms[i][:10]}")

    # Match grayscale target to each channel's histogram
    matched_channels = []
    for i in range(3):  # RGB channels
        # Compute histograms and cumulative distribution functions (CDFs)
        target_hist = cv2.calcHist([target_image], [0], None, [256], [0, 256]).flatten()
        target_hist /= target_hist.sum()  # Normalize histogram
        target_cdf = np.cumsum(target_hist)

        ref_hist = combined_histograms[i]  # Pre-computed average histogram for channel
        ref_hist /= ref_hist.sum()  # Normalize histogram
        ref_cdf = np.cumsum(ref_hist)

        # Debug: Plot CDFs for comparison
        plt.figure()
        plt.plot(target_cdf, label="Target CDF")
        plt.plot(ref_cdf, label="Reference CDF")
        plt.legend()
        plt.title(f"CDF Comparison for Channel {i}")
        plt.show()

        # Create mapping for pixel intensities
        mapping = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for target_idx in range(256):
            while ref_idx < 256 and ref_cdf[ref_idx] < target_cdf[target_idx]:
                ref_idx += 1
            mapping[target_idx] = ref_idx if ref_idx < 256 else 255

        print(f"Mapping for Channel {i}: {mapping}")  # Debug: Print mapping array

        # Apply the mapping to the grayscale target image
        matched_channel = cv2.LUT(target_image, mapping)
        matched_channels.append(matched_channel)

        # Debug: Check output of matched channel
        print(f"Matched Channel {i} Values: {np.unique(matched_channel)}")
        
    # Merge the matched channels to form the final RGB image
    recolored_image = cv2.merge(matched_channels)

    # Debug: Check final image shape
    print(f"Recolored Image Shape: {recolored_image.shape}")  # Should be (height, width, 3)

    # Normalize the recolored image for consistent pixel intensity
    recolored_image = cv2.normalize(recolored_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save the recolored RGB image
    cv2.imwrite(output_path, recolored_image)

    # Debug: Reload and verify the saved image
    loaded_image = cv2.imread(output_path)
    print(f"Saved Image Shape: {loaded_image.shape}")  # Should be (height, width, 3)

    # Visualization for confirmation
    plt.imshow(cv2.cvtColor(recolored_image, cv2.COLOR_BGR2RGB))
    plt.title("Recolored Image")
    plt.show()


def recolor_image_constrast(target_path, reference_paths, output_path):
    """
    Recolor the target image using the histograms of multiple reference images.
    
    Parameters:
    - target_path (str): Path to the target image (background-removed).
    - reference_paths (list[str]): Paths to the reference images (background-removed).
    - output_path (str): Path to save the recolored image.
    """
    
    if os.path.exists(output_path):
        print(f"Recolored image already exists at {output_path}")
        return
    
    print(f"Recoloring image at {target_path} using {len(reference_paths)} reference images")
    # Load the target image
    target_image = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if target_image is None:
        raise ValueError(f"Could not load target image from {target_path}")

    # Load reference images and calculate the average histogram
    combined_histogram = np.zeros((256, 3))  # For RGB channels
    for ref_path in reference_paths:
        reference_image = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if reference_image is None:
            print(f"Skipping invalid reference image at {ref_path}")
            continue

        for i in range(3):  # RGB channels
            ref_hist = cv2.calcHist([reference_image], [i], None, [256], [0, 256])
            combined_histogram[:, i] += ref_hist[:, 0]

    # Normalize the combined histogram
    combined_histogram /= len(reference_paths)

    # Perform histogram matching for each channel
    matched_image = target_image.copy()
    for i in range(3):  # RGB channels
        target_hist = cv2.calcHist([target_image], [i], None, [256], [0, 256])
        target_cdf = np.cumsum(target_hist / target_hist.sum())
        ref_cdf = np.cumsum(combined_histogram[:, i] / combined_histogram[:, i].sum())

        # Create a mapping for pixel intensities
        mapping = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for target_idx in range(256):
            while ref_idx < 256 and ref_cdf[ref_idx] < target_cdf[target_idx]:
                ref_idx += 1
            mapping[target_idx] = ref_idx if ref_idx < 256 else 255

        # Apply the mapping to the target image channel
        matched_image[..., i] = cv2.LUT(target_image[..., i], mapping)

    # Save the recolored image
    cv2.imwrite(output_path, matched_image)
    print(f"Recolored image saved to {output_path}")

def rgb_to_lab(target_path, output_path):
    """
    Convert an RGB image to Lαβ color space using OpenCV.
    """
    # Load the RGB image
    rgb_image = cv2.imread(target_path)
    if rgb_image is None:
        raise ValueError(f"Could not load RGB image from {target_path}")

    # Convert to Lαβ color space
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)

    # Save the Lαβ image
    cv2.imwrite(output_path, lab_image)
    print(f"Lαβ image saved to {output_path}")

def lab_to_rgb(target_path, output_path):
    """
    Convert an Lαβ image to RGB color space using OpenCV.
    """
    # Load the Lαβ image
    lab_image = cv2.imread(target_path)
    if lab_image is None:
        raise ValueError(f"Could not load Lαβ image from {target_path}")

    # Convert to RGB color space
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Save the RGB image
    cv2.imwrite(output_path, rgb_image)
    print(f"RGB image saved to {output_path}")

def recolor_image_lab(target_path, reference_paths, output_path):
    """
    Recolors the grayscale target image based on the histogram of multiple reference RGB images in the Lαβ color space.
    
    Parameters:
    - target_path (str): Path to the grayscale target image.
    - reference_paths (list[str]): List of paths to reference RGB images.
    - output_path (str): Path to save the recolored image.
    """
    # Load the target grayscale image
    target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if target_image is None:
        raise ValueError(f"Could not load target image from {target_path}")
    print(f"Target Image Shape: {target_image.shape}")

    # Ensure dimensions are correct
    height, width = target_image.shape

    # Load and preprocess the reference images
    reference_images = []
    for ref_path in reference_paths:
        ref_image = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref_image is None:
            print(f"Skipping invalid reference image at {ref_path}")
            continue
        # Resize reference images to match the target dimensions
        ref_image = cv2.resize(ref_image, (width, height))
        reference_images.append(ref_image)

    if not reference_images:
        raise ValueError("No valid reference images provided.")
    print(f"Number of valid reference images: {len(reference_images)}")

    # Convert reference images from RGB to Lαβ color space
    reference_lab_images = [cv2.cvtColor(ref_image, cv2.COLOR_BGR2LAB) for ref_image in reference_images]

    # Calculate average histograms for a and b channels across reference images
    combined_histograms_a = np.zeros(256)
    combined_histograms_b = np.zeros(256)
    
    for ref_lab_image in reference_lab_images:
        a_channel = ref_lab_image[:, :, 1].flatten()
        b_channel = ref_lab_image[:, :, 2].flatten()

        # Accumulate histograms
        combined_histograms_a += np.histogram(a_channel, bins=256, range=(-128, 127))[0]
        combined_histograms_b += np.histogram(b_channel, bins=256, range=(-128, 127))[0]

    # Normalize histograms
    combined_histograms_a /= len(reference_images)
    combined_histograms_b /= len(reference_images)

    # Convert target image to Lαβ color space
    target_rgb = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
    target_lab = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2LAB)
    target_l_channel = target_lab[:, :, 0]
    target_a_channel = target_lab[:, :, 1]
    target_b_channel = target_lab[:, :, 2]

    # Match the target a and b channels to the reference histograms
    target_hist_a = np.histogram(target_a_channel.flatten(), bins=256, range=(-128, 127))[0]
    target_hist_b = np.histogram(target_b_channel.flatten(), bins=256, range=(-128, 127))[0]

    # Normalize target histograms
    target_hist_a = target_hist_a / target_hist_a.sum()
    target_hist_b = target_hist_b / target_hist_b.sum()

    # Compute CDFs
    target_cdf_a = np.cumsum(target_hist_a)
    target_cdf_b = np.cumsum(target_hist_b)
    ref_cdf_a = np.cumsum(combined_histograms_a)
    ref_cdf_b = np.cumsum(combined_histograms_b)

    # Create mappings for a and b channels
    mapping_a = np.zeros(256, dtype=np.uint8)
    mapping_b = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        # Find closest matching intensity for a and b channels
        mapping_a[i] = np.searchsorted(ref_cdf_a, target_cdf_a[i]) - 1
        mapping_b[i] = np.searchsorted(ref_cdf_b, target_cdf_b[i]) - 1

    # Apply mappings to the target image's a and b channels
    recolored_a_channel = cv2.LUT(target_a_channel, mapping_a)
    recolored_b_channel = cv2.LUT(target_b_channel, mapping_b)

    # Rebuild the Lαβ image
    recolored_lab_image = np.stack([target_l_channel, recolored_a_channel, recolored_b_channel], axis=-1)

    # Convert back to RGB
    recolored_rgb_image = cv2.cvtColor(recolored_lab_image, cv2.COLOR_LAB2BGR)

    # Save the recolored image
    cv2.imwrite(output_path, recolored_rgb_image)
    print(f"Recolored image saved to {output_path}")












def add_back_background(input_path, background_path, output_path, weight_foreground=0.7, weight_background=0.3):
    """
    Adds the background back to an input image with adjustable blending weights.

    Parameters:
    - input_path (str): Path to the input image (e.g., a decolorized or foreground-only image).
    - background_path (str): Path to the background image.
    - output_path (str): Path to save the output image.
    - weight_foreground (float): Weight for the input image in the blend (default: 0.7).
    - weight_background (float): Weight for the background image in the blend (default: 0.3).
    """
    # Load the input image (foreground or decolorized image)
    input_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        raise ValueError(f"Could not load input image from {input_path}")

    # Load the background image
    background_image = cv2.imread(background_path, cv2.IMREAD_COLOR)
    if background_image is None:
        raise ValueError(f"Could not load background image from {background_path}")

    # Ensure the images have the same dimensions
    if input_image.shape[:2] != background_image.shape[:2]:
        background_image = cv2.resize(background_image, (input_image.shape[1], input_image.shape[0]))

    # Convert the input image to BGR if it's grayscale
    if len(input_image.shape) == 2 or input_image.shape[2] == 1:  # Grayscale input
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    # Blend the input image with the background
    blended_image = cv2.addWeighted(input_image, weight_foreground, background_image, weight_background, 0)

    # Save the output image
    cv2.imwrite(output_path, blended_image)
    print(f"Output image saved to {output_path}")
    