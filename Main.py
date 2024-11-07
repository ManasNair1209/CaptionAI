import os
import pandas as pd
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import gradio 

# Load pre-trained models
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generation settings
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate captions for all images in a folder
def predict_captions_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]  # List of image files
    image_paths = [os.path.join(folder_path, f) for f in image_files]  # Full paths to the images

    # DataFrame to store the results
    results = pd.DataFrame(columns=["File Name", "Caption"])

    # Process each image
    for img_file, img_path in zip(image_files, image_paths):
        i_image = Image.open(img_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        # Extract features and generate captions
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()

        # Add to DataFrame
        results = pd.concat([results, pd.DataFrame([[img_file, caption]], columns=["File Name", "Caption"])], ignore_index=True)
        print(f"Processed: {img_file}, Caption: {caption}")

    return results

# Folder path containing the frames
folder_path = '/content/drive/MyDrive/CaptionerAI/Images'

# Generate captions for all images in the folder
df_captions = predict_captions_from_folder(folder_path)

# Save the captions to an Excel file
output_excel = "captions.xlsx"
df_captions.to_excel(output_excel, index=False)

print(f"Captions saved to {output_excel}")


# Assuming df_captions is already populated

# Sort the DataFrame by the 'File Name' column
df_captions = df_captions.sort_values(by='File Name').reset_index(drop=True)

# Save the sorted captions to an Excel file
output_excel = "/content/drive/MyDrive/CaptionerAI/Captions.xlsx"
df_captions.to_excel(output_excel, index=False)

print(f"Sorted captions saved to {output_excel}")

import cv2
import os
import pandas as pd

# Load the Excel file with frame names and captions
df = pd.read_excel('/content/drive/MyDrive/CaptionerAI/Captions.xlsx')

# Folder containing the frames
image_folder = '/content/drive/MyDrive/CaptionerAI/Images'

# Video settings
output_video = '/content/drive/MyDrive/CaptionerAI/captioned_video.mp4'
frame_width, frame_height = 640, 480  # Adjust depending on your frame size
fps = 24  # Frames per second (adjust based on original video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec

# Initialize the VideoWriter
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Loop through the DataFrame and add captions to each frame
for idx, row in df.iterrows():
    img_file = row['File Name']
    caption = row['Caption']
    img_path = os.path.join(image_folder, img_file)

    # Read the image
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to load image {img_file}")
        continue

    # Resize the image to the desired frame size
    img = cv2.resize(img, (frame_width, frame_height))

    # Define the position for the text (bottom of the image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

    text_x = (img.shape[1] - text_size[0]) // 2  # Center the text horizontally
    text_y = img.shape[0] - 50  # Position the text at the bottom

    # Add a black background for the caption
    cv2.rectangle(img, (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)

    # Put the caption text on the image
    cv2.putText(img, caption, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    # Write the image with the caption into the video
    video_writer.write(img)

# Release the VideoWriter
video_writer.release()

print(f"Video saved as {output_video}")
