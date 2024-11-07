import cv2
import os

def ext_frames(videopath, outputfolder):
    # Open the video file
    cap = cv2.VideoCapture(videopath)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video")
        return
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    # Create output folder if it doesn't exist
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    
    frame_number = 0
    
    # Loop through frames
    while True:
        ret, frame = cap.read()
        
        # Break if no more frames
        if not ret:
            break
        
        # Create the filename for the extracted frame
        frame_filename = os.path.join(outputfolder, f"frame_{frame_number:04d}.png")
        
        # Save the frame as a PNG file
        cv2.imwrite(frame_filename, frame)
        
        # Increment frame number
        frame_number += 1
        print(f"Extracted frame {frame_number} of {total_frames}")
    
    # Release the video capture object
    cap.release()
    print("Done")

# Example usage
videopath = r'C:\Users\MANASNAIR\Desktop\Python_Coding\MY_Projects\CaptionerAI\Demo_Video.mp4'
outputfolder = 'Images'

ext_frames(videopath, outputfolder)
