# Vehicle Damage Detection System 🚗

A Deep Learning-based Vehicle Damage Detection System that uses a fine-tuned **ResNet50 Convolutional Neural Network (CNN)** to classify vehicle images into different damage categories.

The application provides an interactive interface built using **Streamlit**, allowing users to upload vehicle images and receive predicted damage classifications.

## 🎯 Project Overview

The model classifies vehicle images into six categories:

- F_Breakage
- F_Crushed
- F_Normal
- R_Breakage
- R_Crushed
- R_Normal

<p align="center">
  <img src="../app-screenshot.png" width="800">
</p>
## 🧠 Model Architecture

This project uses **ResNet50**, a pre-trained convolutional neural network available in PyTorch's Torchvision library.

Key implementation details:
- Used ImageNet pre-trained ResNet50 weights.
- Froze the initial layers to retain learned features.
- Fine-tuned Layer 4 of ResNet50.
- Replaced the final fully connected layer with a custom classification head.
- Added Dropout (0.2) for regularization.
- Modified the output layer to predict six classes.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow (PIL)

## 📂 Project Structure

```text
Car-Detection-System/
│
├── model/
│   └── saved_model.pth
│
├── streamlit-app/
│   ├── app.py
│   └── model_helper.py
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "Car Detection system - project"
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it:

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

Navigate to the application directory:

```bash
cd streamlit-app
```

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🖥️ Application Workflow

1. Upload a vehicle image.
2. The uploaded image is preprocessed and resized to 224 × 224.
3. The image is normalized using ImageNet mean and standard deviation.
4. The trained ResNet50 model processes the image.
5. The predicted class is displayed in the Streamlit interface.

## 📌 Future Improvements

- Improve classification performance with additional training data.
- Add confidence scores for predictions.
- Enhance the user interface.
- Deploy the application for public access.

## 👩‍💻 Author

**Lakshmipriya**

