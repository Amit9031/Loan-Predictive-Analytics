"""
Quick script to run model training
"""
from model_training import ModelTrainer

if __name__ == "__main__":
    print("Starting model training...")
    print("="*60)
    
    trainer = ModelTrainer('train_u6lujuX_CVtuZ9i (1).csv')
    trainer.train_all()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("You can now run the web application with: streamlit run app.py")
    print("="*60)



