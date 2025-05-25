
# 🧠 Machine Learning & NLP: Predicting Authors from Classic Literature

This project employs machine learning and natural language processing (NLP) to analyze classic literary works and predict the author of a given phrase. By examining textual patterns and stylistic nuances, the model learns to attribute authorship with notable accuracy.

## 📚 Overview

- **Objective**: Develop a model that can predict the author of a text snippet from classic literature.
- **Techniques Used**:
  - Text vectorization and tokenization
  - Sequential modeling with LSTM (Long Short-Term Memory) networks
- **Tools & Libraries**:
  - Python
  - TensorFlow & Keras
  - Pandas & NumPy

## 📁 Repository Structure

- `Text_Author.csv`: Dataset containing text excerpts and corresponding author labels.
- `text-analysis-detect-author-seq-lstm.py`: Python script for data preprocessing, model training, and evaluation.
- `README.md`: Project documentation.
- `LICENSE`: MIT License.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/markiskorova/Machine-Learning-NLP-Predict-Author.git
   cd Machine-Learning-NLP-Predict-Author
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required packages**:

   ```bash
   pip install tensorflow pandas numpy
   ```

### Running the Model

Execute the script to train and evaluate the model:

```bash
python text-analysis-detect-author-seq-lstm.py
```

*The script will process the data, train the LSTM model, and output evaluation metrics.*

## 📊 Dataset Details

- **Source**: Curated collection of classic literary texts.
- **Format**: CSV file with two columns:
  - `text`: Excerpt from a literary work.
  - `author`: Name of the author.

## 🔍 Model Architecture

- **Embedding Layer**: Converts words into vector representations.
- **LSTM Layer**: Captures sequential dependencies in the text.
- **Dense Output Layer**: Outputs probabilities for each author class.

## 📈 Evaluation Metrics

- **Accuracy**: Measures the proportion of correct predictions.
- **Loss**: Evaluates the model's prediction error.

## 🛠️ Future Enhancements

- Incorporate more diverse literary works to improve model generalization.
- Experiment with advanced architectures like Bidirectional LSTMs or Transformers.
- Implement a user interface for interactive author prediction.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact the repository maintainer.
