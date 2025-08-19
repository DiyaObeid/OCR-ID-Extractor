from ArabicOcr import arabicocr
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to process OCR on a single image
def process_image(image_path, out_image, results_list):
    results = arabicocr.arabic_ocr(image_path, out_image)
    print(f"Processed: {image_path}")

    # Save recognized text and bounding box details to the results list
    for result in results:
        bbox, text, confidence = result
        results_list.append({
            "Image": image_path,
            "Text": text,
            "Confidence": confidence,
            "Bounding Box": bbox
        })

    # Display the image with Matplotlib
    img = cv2.imread(out_image, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(f'processed_{os.path.basename(out_image)}', img)  # Save processed output

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"OCR Output: {os.path.basename(image_path)}")
    plt.show()


# ---------- MAIN SCRIPT ----------
results_list = []

# 👉 Change this to your actual image path
image_path = r"D:\Projects\OCRtask\ID-front.jpeg"
out_image = r"D:\Projects\OCRtask\ID-front.jpeg"

# Check if the image exists
if os.path.exists(image_path):
    process_image(image_path, out_image, results_list)

    # Save results to Excel
    results_df = pd.DataFrame(results_list)
    excel_file = "ocr_results.xlsx"
    results_df.to_excel(excel_file, index=False)
    print(f"OCR results saved to {excel_file}")
else:
    print(f"Image not found: {image_path}")
