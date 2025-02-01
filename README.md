# Self-Driving Car Simulation

This project simulates a self-driving car using a deep learning model to predict steering angles based on input images from a car's camera. The model is trained using a modified NVIDIA architecture.

## Project Structure

- `car_simulation.ipynb`: Jupyter notebook for data preprocessing, model training, and evaluation.
- `main.py`: Python script to run the self-driving car simulation using Flask and SocketIO.
- `model/`: Directory containing the trained model file `model.h5`.

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/arjun-venugopal/Self-Driving.git
    cd self-driving
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset:**
    - Ensure the dataset is downloaded and extracted to the appropriate directory as specified in the notebook.

4. **Train the model:**
    - Open `car_simulation.ipynb` and run all cells to preprocess data, train the model, and save the trained model to `model/model.h5`.

## Running the Simulation

1. **Start the Flask-SocketIO server:**
    ```bash
    python main.py
    ```

2. **Connect the simulator:**
    - Open your self-driving car simulator and connect it to the server running on `localhost:4567`.

## Udacity Car Simulator

This project uses the Udacity car simulator to test the self-driving car model. You can download the simulator from the [Udacity GitHub repository](https://github.com/udacity/self-driving-car-sim).

## Model Architecture

The model is based on a modified NVIDIA architecture with the following layers:
- **Convolutional Layers**: Extract features from the input images.
- **Flatten Layer**: Flatten the 3D outputs to 1D.
- **Fully Connected Layers**: Perform the final steering angle prediction with ELU activation and Dropout for regularization.

## Image Preprocessing

The images are preprocessed by:
- **Cropping**: Remove unnecessary parts of the image.
- **Color Space Conversion**: Convert the image to YUV color space.
- **Gaussian Blur**: Apply Gaussian blur to reduce noise.
- **Resizing**: Resize the image to 200x66 pixels.
- **Normalization**: Normalize the image pixel values.

## Custom Metrics

A custom Mean Squared Error (MSE) metric is used for model evaluation.

## References

- **main.py**: This script sets up a Flask-SocketIO server to communicate with the Udacity car simulator. It handles image preprocessing, model prediction, and sending control commands to the simulator.
- **car_simulation.ipynb**: This Jupyter notebook contains the code for data preprocessing, model training, and evaluation. It includes steps for augmenting the dataset, defining the model architecture, and training the model.

## License

This project is licensed under the MIT License.