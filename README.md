## Project Overview

This project aims to classify video game genres using a neural network model. The dataset consists of attributes related to video games such as platform, developer, scores, and player feedback. The project includes data preprocessing, feature encoding, building a neural network model using TensorFlow, and evaluating the model's performance.

### Key Features
- Data preprocessing, including handling missing values and encoding categorical features.
- Building a neural network model with TensorFlow for genre classification.
- Visualizations of model performance and feature correlations.
- Saving the trained model for future use.

## Dataset

The dataset used is `games-data.csv`, which includes the following key features:
- `platform`: Platform on which the game is released.
- `developer`: Developer of the game.
- `score`, `user score`: Ratings by critics and users.
- `players`: Number of players (categorical).
- `genre`: The genre(s) of the game (target variable).

The target variable (`primary_genre`) is derived by extracting the primary genre from the `genre` column.

## Project Structure

```
├── Video_Game_Genre_Classification_Using_Neural_Networks_Main.ipynb
├── games-data.csv
└── game_genre_classifier.h5
```

## Installation

To run this project, install the following Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Running the Project

### Local Setup

1. Ensure that all dependencies are installed.
2. Place `games-data.csv` in the project directory.
3. Open the Jupyter notebook and run the cells sequentially.

### Google Colab Setup

1. Upload `games-data.csv` to Colab or link your dataset from Google Drive.
2. Install required libraries:
   ```python
   !pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```
3. Load the dataset using the appropriate file path (e.g., `/content/games-data.csv`).
4. Run all cells to execute the project.

## Model Architecture

The neural network model uses TensorFlow and has the following structure:
- **Input Layer**: Based on the number of features.
- **Hidden Layers**: Two dense layers with 64 and 32 neurons, ReLU activation.
- **Output Layer**: Softmax activation for multi-class classification.

Model compilation:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## Evaluation

The model is evaluated using:
- Confusion Matrix
- Classification Report
- Accuracy and Loss Curves

These metrics demonstrate the model's ability to predict the primary genre of video games.

## Recommendations

- **Feature Selection**: Add more features like game release date or genre popularity trends.
- **Data Augmentation**: Explore techniques like SMOTE to handle class imbalances.
- **Hyperparameter Tuning**: Experiment with different architectures, optimizers, and learning rates.
- **Model Deployment**: Use `game_genre_classifier.h5` for inference in a web application or API.

## Saving and Loading the Model

The trained model is saved as `game_genre_classifier.h5`. To load the model:

```python
from tensorflow.keras.models import load_model
model = load_model("game_genre_classifier.h5")
```

## Visualizations

The following visualizations are included:
- **Correlation Heatmap**: Displays relationships between numerical features.
- **Accuracy and Loss Curves**: Shows model performance during training.
- **Confusion Matrix**: Illustrates the classification performance for each genre.

## Contributing

Feel free to fork the repository and submit a pull request for any contributions.

## License

This project is licensed under the MIT License.
