# Experiment 5.3: Sequence Text Classification using LSTM

## Objective
To classify text sequences using an LSTM-based deep learning model. This experiment demonstrates how to preprocess text data, build a neural network using LSTM, and evaluate its performance on a real-world dataset.

## Dataset
**SMS Spam Collection Dataset**  
- Total samples: 5,572  
- Classes: `ham` (non-spam) and `spam`

## Steps Performed

### 1. Data Loading and Exploration
- Loaded and inspected the SMS dataset.
- Displayed initial rows and performed basic statistical exploration.
- Visualized `ham` and `spam` messages using WordClouds.
- Countplots were generated before and after downsampling to balance the dataset.

### 2. Data Preprocessing
- Downsampled `ham` messages to match the number of `spam` messages.
- Added text length as a feature.
- Converted class labels into binary values: `ham` → 0, `spam` → 1.

### 3. Tokenization and Padding
- Used Keras `Tokenizer` with:
  - `vocab_size = 500`
  - `max_len = 50`
- Performed text-to-sequence conversion and padded sequences.

### 4. Model Architecture
LSTM model built using Keras:
```python
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
```

### 5. Training
- Trained over 30 epochs with early stopping.
- Achieved good convergence within 9 epochs.

### 6. Evaluation Metrics
- **Final Training Accuracy**: 97.57%
- **Final Validation Accuracy**: 93.98%
- Metrics used: Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualized for performance validation.

### 7. Visualization
- Training vs. Validation Accuracy and Loss curves were plotted.

## Results

| Metric           | Value     |
|------------------|-----------|
| Train Accuracy   | 97.57%    |
| Validation Accuracy | 93.98% |
| Final Loss (Val) | 0.1504    |

## Conclusion
The LSTM model effectively classified spam and ham messages with high accuracy, proving LSTM's capabilities in sequence-based NLP tasks. Further improvements can be achieved with:
- Hyperparameter tuning
- Using pre-trained embeddings (e.g., GloVe)
- Larger vocab size
