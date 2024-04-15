import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import numpy as np
import os
from datetime import datetime
import json

# Global variables for account management
account_data = {}
current_user = None

# Global variables for image and filter management
img = None
is_video_stream_running = False
cap = None

# Filter toggles
blur_faces = False
use_grayscale = False
use_red_channel = False
use_green_channel = False
use_blue_channel = False
use_sepia_tone = False
use_cartoonize = False
use_change_brightness = False
use_contrast_adjustment = False
use_noise_reduction = False
use_edge_detection = False
use_zoom_in_out = False
use_rotation = False

# Function to save account data to a JSON file
def save_account_data():
    with open('account_data.json', 'w') as f:
        json.dump(account_data, f)

# Load or initialize account data
def load_account_data():
    global account_data
    try:
        with open('account_data.json', 'r') as f:
            account_data = json.load(f)
    except FileNotFoundError:
        account_data = {}


# Function to open file dialog
def open_file_dialog():
    filename = filedialog.askopenfilename()
    print("Selected file:", filename)  # Here you can load the selected file into OpenCV or any other processing

# Function to create account
def create_account():
    global account_data
    username = username_entry.get()
    password = password_entry.get()
    if username and password:
        # Create account folder
        account_folder = os.path.join("accounts", username)
        if not os.path.exists(account_folder):
            os.makedirs(account_folder)
            account_data[username] = {"password": password, "folder": account_folder}
            save_account_data()  # Save account data
            messagebox.showinfo("Success", "Account created successfully!")
        else:
            messagebox.showerror("Error", "Account already exists!")
    else:
        messagebox.showerror("Error", "Username and password cannot be empty!")

# Function to log in
def login():
    global current_user
    load_account_data()  # Load account data
    username = username_entry.get()
    password = password_entry.get()
    if username in account_data and account_data[username]["password"] == password:
        current_user = username
        messagebox.showinfo("Success", "Login successful!")
        photo_button.config(state="normal")
    else:
        messagebox.showerror("Error", "Invalid username or password!")

# Function to take a photo
def take_photo():
    global img
    if isinstance(img, np.ndarray):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(account_data[current_user]["folder"], f"photo_{timestamp}.jpg")
        cv2.imwrite(img_path, img)
        messagebox.showinfo("Success", "Photo taken and saved successfully!")

# Function to save photo in the user's account folder
def save_photo_in_folder():
    global img, current_user
    if current_user:
        if isinstance(img, np.ndarray):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(account_data[current_user]["folder"], f"photo_{timestamp}.jpg")
            cv2.imwrite(img_path, img)
            messagebox.showinfo("Success", "Photo saved successfully!")
    else:
        messagebox.showerror("Error", "Please log in first!")


# Image processing functions
def apply_filters():
    global img
    if img is None:
        return

    if use_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif use_red_channel:
        img[:, :, 0] = 0  # Set blue channel to zero
        img[:, :, 1] = 0  # Set green channel to zero
    elif use_green_channel:
        img[:, :, 0] = 0  # Set blue channel to zero
        img[:, :, 2] = 0  # Set red channel to zero
    elif use_blue_channel:
        img[:, :, 1] = 0  # Set green channel to zero
        img[:, :, 2] = 0  # Set red channel to zero
    elif use_sepia_tone:
        img = apply_sepia_tone(img)
    elif use_cartoonize:
        img = apply_cartoonize(img)
    # Add other filters as elif blocks...

# Function to toggle face blur option
def toggle_face_blur():
    global blur_faces
    blur_faces = not blur_faces

# Function to toggle grayscale filter
def toggle_grayscale():
    global use_grayscale
    use_grayscale = not use_grayscale

# Function to toggle individual color channels
def toggle_red_channel():
    global use_red_channel
    use_red_channel = not use_red_channel

def toggle_green_channel():
    global use_green_channel
    use_green_channel = not use_green_channel

def toggle_blue_channel():
    global use_blue_channel
    use_blue_channel = not use_blue_channel

# Function to toggle sepia tone filter
def toggle_sepia_tone():
    global use_sepia_tone
    use_sepia_tone = not use_sepia_tone

# Function to toggle cartoonize filter
def toggle_cartoonize():
    global use_cartoonize
    use_cartoonize = not use_cartoonize

#Function to toggle change brightness filter
def toggle_change_brightness():
    global use_change_brightness
    use_change_brightness = not use_change_brightness

#Function to toggle contrast adjustment filter 
def toggle_contrast_adjustment():
    global use_contrast_adjustment
    use_contrast_adjustment = not use_contrast_adjustment

#Function to toggle noise reduction filter 
def toggle_noise_reduction():
    global use_noise_reduction
    use_noise_reduction = not use_noise_reduction

#Function to toggle edge detection filter 
def toggle_edge_detection():
    global use_edge_detection
    use_edge_detection = not use_edge_detection

#Function to toggle zoom in/out filter 
def toggle_zoom_in_out():
    global use_zoom_in_out
    use_zoom_in_out = not use_zoom_in_out

#Function to toggle rotation filter 
def toggle_rotation():
    global use_rotation
    use_rotation = not use_rotation

# Function to apply vhange brightness filter 
def change_brightness(img):
    value=50 # we've defined the value throgh which the brightness should be changed
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# Function to apply contrast filter by pre-defined factor
def change_contrast(img, factor=3):
    alpha = factor  # Simple contrast control
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return adjusted

#Function to apply noise reduction filter 
def reduce_noise(img):
    return cv2.medianBlur(img, 5)

#Function to apply edge detection
def edge_detection(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert edges to a 3 channel image to match original format
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_colored

#Function to apply zoom in/out filter
def zoom_image(img, scale=0.5):
    height, width = img.shape[:2]
    new_height, new_width = int(scale * height), int(scale * width)
    return cv2.resize(img, (new_width, new_height))

#Function to apply rotation filter by an angle of 90
def rotate_image(img, angle=90):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (width, height))
    return rotated


# Function to apply sepia tone filter
def apply_sepia_tone(img):
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_img = cv2.transform(img, sepia_matrix)
    return sepia_img

# Function to apply cartoonize filter
def apply_cartoonize(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median blur to smoothen the image
    smooth_gray = cv2.medianBlur(gray, 5)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(smooth_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    # Create a color version of the image
    color = cv2.bilateralFilter(img, 9, 300, 300)

    # Combine the edges and color image to create a cartoon effect
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

# Function to start the video stream with face blur option
def start_video_stream():
    global img, cap
    try:
        # Load the required trained XML classifiers
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Capture frames from a camera
        cap = cv2.VideoCapture(0)

        # Loop runs if capturing has been initialized.
        while True:
            # Reads frames from a camera
            ret, img = cap.read()

            if blur_faces:
                # Convert the image to gray scale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces of different sizes in the input image
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # Extract the region of interest (ROI) which is the face area
                    roi = img[y:y+h, x:x+w]

                    # Apply Gaussian blur to the ROI
                    blurred_roi = cv2.GaussianBlur(roi, (25, 25), 0)

                    # Replace the original face region with the blurred region
                    img[y:y+h, x:x+w] = blurred_roi

            if use_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif use_red_channel:
                img[:, :, 0] = 0  # Set blue channel to zero
                img[:, :, 1] = 0  # Set green channel to zero
            elif use_green_channel:
                img[:, :, 0] = 0  # Set blue channel to zero
                img[:, :, 2] = 0  # Set red channel to zero
            elif use_blue_channel:
                img[:, :, 1] = 0  # Set green channel to zero
                img[:, :, 2] = 0  # Set red channel to zero
            elif use_sepia_tone:
                img = apply_sepia_tone(img)
            elif use_cartoonize:
                img = apply_cartoonize(img)
            # In the main video stream loop, add:
            elif use_edge_detection:
                img = edge_detection(img)
            elif use_change_brightness:
                img = change_brightness(img)
            elif use_rotation:
                img = rotate_image(img)
            elif use_zoom_in_out:
                img = zoom_image(img)
            elif use_contrast_adjustment:
                img = change_contrast(img)
            elif use_noise_reduction:
                img = reduce_noise(img)


            # Apply selected filters
            apply_filters()

            # Display the resulting image
            cv2.imshow('Video Stream', img)

            # Break the loop if the Esc key is pressed
            if cv2.waitKey(30) & 0xff == 27:
                break

        # Release the video capture object
        cap.release()

        # Close all windows
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)

# Function to start the video stream in a separate thread
def start_video_stream_thread():
    video_stream_thread = threading.Thread(target=start_video_stream)
    video_stream_thread.start()

# Create main window
root = tk.Tk()
root.title("Account Database and Image Processing")

# Create username label and entry
username_label = tk.Label(root, text="Username:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

# Create password label and entry
password_label = tk.Label(root, text="Password:")
password_label.pack()
password_entry = tk.Entry(root, show="*")
password_entry.pack()

# Create buttons for account actions
create_button = tk.Button(root, text="Create Account", command=create_account)
create_button.pack()
login_button = tk.Button(root, text="Login", command=login)
login_button.pack()

# Initialize account data
load_account_data()

# Load account data when the application starts
load_account_data()

# Create menu
menu = tk.Menu(root)
root.config(menu=menu)

# Create file menu
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=open_file_dialog)

# Create filters menu
filters_menu = tk.Menu(menu)
menu.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_checkbutton(label="Blur Faces", command=toggle_face_blur)
filters_menu.add_separator()
filters_menu.add_checkbutton(label="Grayscale", command=toggle_grayscale)
filters_menu.add_separator()
filters_menu.add_checkbutton(label="Red Channel", command=toggle_red_channel)
filters_menu.add_checkbutton(label="Green Channel", command=toggle_green_channel)
filters_menu.add_checkbutton(label="Blue Channel", command=toggle_blue_channel)
filters_menu.add_separator()
filters_menu.add_checkbutton(label="Sepia Tone", command=toggle_sepia_tone)
filters_menu.add_checkbutton(label="Cartoonize", command=toggle_cartoonize)
filters_menu.add_checkbutton(label="Change Brightness", command=toggle_change_brightness)
filters_menu.add_checkbutton(label="Contrast Adjustment", command=toggle_contrast_adjustment)
filters_menu.add_checkbutton(label="Noise Reduction", command=toggle_noise_reduction)
filters_menu.add_checkbutton(label="Edge Detection", command=toggle_edge_detection)
filters_menu.add_checkbutton(label="Zoom In/Out", command=toggle_zoom_in_out)
filters_menu.add_checkbutton(label="Rotation", command=toggle_rotation)

# Create take photo button
photo_button = tk.Button(root, text="Take Photo", command=take_photo, state="disabled")
photo_button.pack(side="left", padx=20, pady=10)

# Create start/stop video stream button
video_stream_button = tk.Button(root, text="Start/Stop Video Stream", command=start_video_stream_thread)
video_stream_button.pack(side="right", padx=20, pady=10)

# Run the Tkinter event loop
root.mainloop()
