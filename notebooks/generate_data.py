import os
import cv2
import numpy as np

def generate_synthetic_dataset(base_path="dataset", num_samples=10):
    folders = ["train/fractured", "train/normal", "test/fractured", "test/normal"]
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
        for i in range(num_samples):
            # Generate a random black & white image to simulate X-ray
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            # Add a white line to simulate a "fracture" in fractured folder
            if "fractured" in folder:
                cv2.line(img, (50, 50), (170, 170), (255, 255, 255), 2)
            cv2.imwrite(os.path.join(base_path, folder, f"img_{i}.png"), img)
    print(f"Generated synthetic dataset with {num_samples} samples per class in {base_path}.")

if __name__ == "__main__":
    generate_synthetic_dataset()
