# Brain-Tumor-Detection-System
This project implements a Deep Learning model to detect Brain Tumors from MRI images using Transfer Learning (MobileNetV2). It includes a Streamlit web application for easy interaction.

## Project Structure
- `train_model.py`: Script to train the CNN model. Includes dummy data generation for testing.
- `app.py`: Streamlit web application for deployment.
- `requirements.txt`: List of dependencies.
- `dataset/`: Directory where training data is stored (can be auto-generated for testing).
- `model.h5`: The trained model file (generated after running `train_model.py`).

## Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset (Important!)**
    - The script includes a dummy data generator that will create a fake dataset if none exists, just to prove the pipeline works.
    - **For Real Training**: Download a Brain Tumor MRI dataset (e.g., from [Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)) and extract it into the `dataset/` folder.
    - Structure should be:
        ```
        dataset/
            yes/  (Tumor images)
            no/   (No Tumor images)
        ```

3.  **Train the Model**
    Run the training script to generate `model.h5`.
    ```bash
    python train_model.py
    ```
    *Note: If no dataset is present, it will generate random noise images for testing.*

4.  **Run the Application**
    Start the Streamlit app.
    ```bash
    streamlit run app.py
    ```

## Deployment on Streamlit Cloud

1.  Push this entire folder to a GitHub repository.
2.  Go to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Connect your GitHub and select the repository.
4.  Select `app.py` as the main file.
5.  Click **Deploy**.
