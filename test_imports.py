"""Test all imports"""
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy error: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__}")
except Exception as e:
    print(f"✗ Pandas error: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn {sklearn.__version__}")
except Exception as e:
    print(f"✗ scikit-learn error: {e}")

try:
    import streamlit
    print(f"✓ Streamlit {streamlit.__version__}")
except Exception as e:
    print(f"✗ Streamlit error: {e}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"⚠ TensorFlow error: {e} (neural networks may not work)")

print("\nAll critical imports successful!")






