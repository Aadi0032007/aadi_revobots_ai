# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:11:02 2026

@author: Aadi
"""

import os
import json
import textwrap
import numpy as np
import cv2
from PIL import Image
from google import genai
from google.genai import types

# --- User Provided Marker Function ---
def put_the_marker(
    image: np.ndarray,
    coords: tuple,
    radius=15,
    border_color=(255, 0, 0),  # Note: In RGB, this is Blue. In BGR, this is Red.
    cross_color=(255, 0, 0),   
    bg_color=(255, 255, 255),  # white fill
):
    """
    Draw a marker on the given image at the specified coords.
    """
    if coords is None:
        return image  # no marker

    x, y = int(coords[0]), int(coords[1])
    
    # Circle background
    cv2.circle(image, (x, y), radius, bg_color, -1)
    # Circle border
    cv2.circle(image, (x, y), radius, border_color, 2)
    # Cross inside
    cv2.line(image, (x, y - (radius - 1)), (x, y + (radius - 1)), cross_color, 2)
    cv2.line(image, (x - (radius - 1), y), (x + (radius - 1), y), cross_color, 2)
    return image

# ==========================================
# API 1: Get Coordinates from Gemini
# ==========================================
def get_object_coordinates(image_input, prompt, api_key=None):
    """
    Detects objects and returns their coordinates (bounding boxes and centroids).
    """
    # 1. Setup Client
    os.environ["GOOGLE_API_KEY"] = "" # Revolabs
    
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("API Key not found. Set GOOGLE_API_KEY env var.")
    
    client = genai.Client(api_key=key)
    MODEL_ID = "gemini-robotics-er-1.5-preview"
    # MODEL_ID = "gemini-3-flash-preview"
    # MODEL_ID = 'gemini-2.5-flash-lite'

    # Handle Input (Ensure PIL for API)
    if isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input)
    else:
        pil_img = image_input

    # Resize for API
    base_width = 800
    w_percent = (base_width / float(pil_img.size[0]))
    h_size = int((float(pil_img.size[1]) * float(w_percent)))
    pil_img_resized = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)

    # Prompt
    system_instruction = textwrap.dedent("""\
        Detect the objects specified in the user task.
        Return a JSON list. Each item must have:
        - "label": The name of the object.
        - "box_2d": [ymin, xmin, ymax, xmax] coordinates normalized to 0-1000.
    """)
    full_prompt = f"{system_instruction}\n\nUser Task: {prompt}"

    # Call Gemini
    config = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(
          thinking_budget=0
        )
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[pil_img_resized, full_prompt],
            config=config,
        )
        text_resp = response.text
        if "```json" in text_resp:
            text_resp = text_resp.split("```json")[1].split("```")[0]
        data = json.loads(text_resp)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return []

    # Process Coordinates
    orig_width, orig_height = pil_img.size
    results = []
    for item in data:
        if "box_2d" in item:
            ymin, xmin, ymax, xmax = item["box_2d"]
            
            abs_y1 = int(ymin / 1000 * orig_height)
            abs_x1 = int(xmin / 1000 * orig_width)
            abs_y2 = int(ymax / 1000 * orig_height)
            abs_x2 = int(xmax / 1000 * orig_width)
            
            cx = int((abs_x1 + abs_x2) / 2)
            cy = int((abs_y1 + abs_y2) / 2)

            results.append({
                "label": item.get("label", "object"),
                "box": [abs_y1, abs_x1, abs_y2, abs_x2],
                "centroid": (cx, cy)
            })
            
    return results

# --- API 2: Draw Markers (Modified) ---
def draw_markers_on_image(image_input, detections):
    """
    Draws bounding boxes and the specific marker on the image.
    NO LABELS are drawn.
    """
    # Ensure numpy array
    if isinstance(image_input, Image.Image):
        img = np.array(image_input)
    else:
        img = image_input.copy()

    # If image is RGB (from PIL), colors (0,0,255) will appear Blue.
    # If image is BGR (from OpenCV), colors (0,0,255) will appear Red.
    
    for item in detections:
        ymin, xmin, ymax, xmax = item["box"]
        centroid = item["centroid"]
        
        # # 1. Draw Bounding Box (Green, thickness 2)
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 2. Draw Marker (Centroid) using the specific function
        put_the_marker(img, centroid)

    return img