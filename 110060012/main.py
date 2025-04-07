import os
from processing import *

# Main directory containing all animal subdirectories
main_directory = 'oregon_wildlife'
removed_color_directory = 'output/recolored_images'
decolor_directory = 'output/decolored_images'
remove_background_directory = 'output/background_removed'
background_directory = 'output/background'
result_directory = 'output/results'
lab_directory = 'output/lab'

# Ensure the output directory and decolor directory exist
os.makedirs(removed_color_directory, exist_ok=True)
os.makedirs(decolor_directory, exist_ok=True)
os.makedirs(remove_background_directory, exist_ok=True)
os.makedirs(background_directory, exist_ok=True)
os.makedirs(result_directory, exist_ok=True)
os.makedirs(lab_directory, exist_ok=True)

# Iterate through each subdirectory (animal) in the main directory
for animal_dir in os.listdir(main_directory):
    animal_dir_path = os.path.join(main_directory, animal_dir)

    # Check if it's a directory and not a file
    if os.path.isdir(animal_dir_path):
        # Ensure the output directories for the current animal exist
        os.makedirs(f"{removed_color_directory}/{animal_dir}", exist_ok=True)
        os.makedirs(f"{decolor_directory}/{animal_dir}", exist_ok=True)
        os.makedirs(f"{remove_background_directory}/{animal_dir}", exist_ok=True)
        os.makedirs(f"{background_directory}/{animal_dir}", exist_ok=True)
        os.makedirs(f"{result_directory}/{animal_dir}", exist_ok=True)
        os.makedirs(f"{lab_directory}/{animal_dir}", exist_ok=True)

        # Step 1: Remove backgrounds from all images and collect paths and decolorize them
        background_removed_images = []
        for file_name in os.listdir(animal_dir_path):
            target_image_path = os.path.join(animal_dir_path, file_name)
            remove_background_path = os.path.join(f"{remove_background_directory}/{animal_dir}", file_name)
            background_path = os.path.join(f"{background_directory}/{animal_dir}", file_name)
            decolor_image_path = os.path.join(f"{decolor_directory}/{animal_dir}", file_name)
            
            try:
                remove_background(
                    input_path=target_image_path,
                    remove_background_path=remove_background_path,
                    background_path=background_path
                )
                decolor_image(
                    input_path=target_image_path,
                    output_path=decolor_image_path,
                    remove_background_path=remove_background_path,
                    background_path=background_path
                )
                background_removed_images.append(remove_background_path)
            except Exception as e:
                print(f"An error occurred while removing background for {file_name}: {e}")

        # Ensure we have some background-removed images to work with
        if not background_removed_images:
            print(f"No valid images for processing in {animal_dir_path}")
            continue
        
        
        # Step 2: Recolor each image using all other images as references
        print(f"Recoloring images in {animal_dir}...")
        for target_image_path in background_removed_images:
            target_file_name = os.path.basename(target_image_path)
            original_image_path = os.path.join(animal_dir_path, target_file_name)
            output_image_path = os.path.join(f"{removed_color_directory}/{animal_dir}", target_file_name)
            decolored_image_path = os.path.join(f"{decolor_directory}/{animal_dir}", target_file_name)
            result_image_path = os.path.join(f"{result_directory}/{animal_dir}", target_file_name)
            background_path = os.path.join(f"{background_directory}/{animal_dir}", target_file_name)
            background_removed_path = os.path.join(f"{remove_background_directory}/{animal_dir}", target_file_name)
            lab_image_path = os.path.join(f"{lab_directory}/{animal_dir}", target_file_name)
            
            # Select all other images as references (excluding the target image itself)
            reference_images = [ref for ref in background_removed_images if ref != target_image_path]

            if not reference_images:
                print(f"No references available for {target_file_name} in {animal_dir}")
                continue

            try:
                # # Ground truth
                # recolor_image_constrast(
                #     target_path=target_image_path,
                #     reference_paths=reference_images,  # Pass all references
                #     output_path=result_image_path
                # )
                # Decolored
                
                recolor_image(
                    target_path=decolored_image_path,
                    reference_paths=reference_images,  # Pass all references
                    output_path=output_image_path
                )
                
                recolor_image_lab(
                    target_path=decolored_image_path,
                    reference_paths=reference_images,  # Pass all references
                    output_path=lab_image_path
                )
            except Exception as e:
                print(f"An error occurred while recoloring {target_file_name}: {e}")
