import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Load trained model
model = load_model("skin_cancer_model.h5")

# Function to classify image
def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Load and preprocess image
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    result = "Malignant" if prediction > 0.5 else "Benign"
    color = "red" if result == "Malignant" else "green"

    # Display image
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Update result label
    label_result.config(text=f"Result: {result}", fg=color, font=("Arial", 16, "bold"))

    # Update preventive measure
    tip_text = "Always use sunscreen & check your skin regularly! ☀️" if result == "Benign" else "Consult a dermatologist immediately! 🔬"
    label_tip.config(text=tip_text, fg="blue", font=("Arial", 12, "italic"))

# GUI Setup
root = tk.Tk()
root.title("Skin Cancer Detection")
root.geometry("420x550")
root.configure(bg="#f4f4f4")

# Motivational Quote
quote = Label(root, text="🌿 \"Early detection saves lives!\" 🌿", font=("Arial", 14, "bold"), bg="#f4f4f4", fg="darkblue")
quote.pack(pady=10)

# Create a stylish frame
frame = Frame(root, bg="white", padx=10, pady=10, relief="ridge", bd=2)
frame.pack(pady=10, padx=20, fill="both", expand=True)

# Upload button
btn_upload = Button(frame, text="Upload Image", command=classify_image, 
                     font=("Arial", 12, "bold"), bg="#008CBA", fg="white", padx=10, pady=5)
btn_upload.pack(pady=10)

# Image display panel
panel = Label(frame, bg="white", relief="sunken", bd=2, width=224, height=224)
panel.pack(pady=10)

# Result label
label_result = Label(frame, text="", font=("Arial", 14, "bold"), bg="white")
label_result.pack(pady=10)

# Preventive measure
label_tip = Label(frame, text="", font=("Arial", 12, "italic"), bg="white", fg="blue")
label_tip.pack(pady=5)

# Run the GUI
root.mainloop()
