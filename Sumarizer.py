import pandas as pd
import re
from transformers import pipeline

# Step 1: Load the Excel Data (Frame-level captions)
def load_excel_data(file_path):
    df_captions = pd.read_excel(file_path)
    return df_captions

# Step 2: Clean and standardize the captions
def clean_caption(caption):
    caption = caption.lower()  # Convert to lowercase
    caption = re.sub(r'[^\w\s]', '', caption)  # Remove punctuation
    caption = re.sub(r'\s+', ' ', caption).strip()  # Remove extra whitespace
    return caption

def preprocess_captions(df):
    df['Cleaned Caption'] = df['Caption'].apply(clean_caption)
    return df

# Step 3: Aggregate captions into a single string
def aggregate_captions(df):
    all_captions = ' '.join(df['Cleaned Caption'].tolist())
    return all_captions

# Step 4: Summarize the aggregated captions
def summarize_captions(captions, max_length=50, min_length=25):
    summarizer = pipeline("summarization")  # Load the summarizer
    summary = summarizer(captions, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Step 5: Execute the entire process
def generate_video_caption(excel_file_path):
    # Load data
    df = load_excel_data(excel_file_path)
    print("Loaded frame-level captions.")
    
    # Preprocess captions
    df = preprocess_captions(df)
    print("Cleaned captions.")
    
    # Aggregate captions
    aggregated_captions = aggregate_captions(df)
    print("Aggregated captions: ", aggregated_captions[:200], "...")  # Displaying only first 200 characters
    
    # Summarize captions
    final_caption = summarize_captions(aggregated_captions)
    print("Final summarized caption: ", final_caption)
    
    return final_caption

# Example usage:
excel_file_path = r'C:\Users\MANASNAIR\Desktop\Python_Coding\MY_Projects\CaptionerAI\Captions.xlsx'
final_caption = generate_video_caption(excel_file_path)
