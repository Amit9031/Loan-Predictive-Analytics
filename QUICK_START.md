# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Models
Run the training script to train all models:
```bash
python model_training.py
```

Or use the simplified runner:
```bash
python run_training.py
```

**Expected Output:**
- Data preprocessing
- Training of all models (Regression, Classification, Clustering, Neural Networks)
- Model evaluation and comparison
- Saved model files (.pkl and .h5)

**Time:** This may take 5-10 minutes depending on your system.

### 3. Launch Web Application
Once training is complete, start the web interface:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 4. Use the Web Interface

#### Model Comparison Page
- View all trained models ranked by accuracy
- See which models perform best
- Visualize model performance

#### Make Prediction Page
- Enter loan applicant details
- Get predictions from the top 2 best models
- See confidence scores
- View feature importance

#### Model Performance Page
- Detailed metrics for each model
- Confusion matrices
- All evaluation metrics

#### Dataset Info Page
- Dataset statistics
- Missing value analysis
- Data distribution

## Troubleshooting

### Issue: Models not found
**Solution:** Make sure you've run `model_training.py` first to generate the model files.

### Issue: TensorFlow/Keras errors
**Solution:** If neural network training fails, the other models will still work. You can comment out the neural network section in `model_training.py` if needed.

### Issue: Streamlit not found
**Solution:** Install Streamlit: `pip install streamlit`

### Issue: Memory errors
**Solution:** Reduce batch size in neural network training or reduce dataset size for testing.

## Best Models

The system automatically selects the top 2 models based on accuracy. These are used for predictions in the web interface.

## File Structure After Training

After running training, you should see:
- `scaler.pkl` - Feature scaler
- `label_encoders.pkl` - Categorical encoders
- `feature_names.pkl` - Feature names
- `*.pkl` - Trained models (one for each algorithm)
- `*.h5` - Neural network models
- `model_results.json` - Evaluation results
- `pca.pkl` - PCA transformer (if used)

## Next Steps

1. Experiment with different hyperparameters
2. Try feature engineering
3. Deploy to cloud (Streamlit Cloud, Heroku, etc.)
4. Add more models or techniques








