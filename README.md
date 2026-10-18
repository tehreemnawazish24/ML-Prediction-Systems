# Assignment 2 — End-to-End Machine Learning System

## Included files
- `Assignment_2_End_to_End_ML_System.ipynb`: complete Jupyter notebook for both tasks.
- `app.py`: Streamlit interface for the two saved models.
- `requirements.txt`: dependencies.
- `artifacts/`: created by running the notebook; contains trained model pipelines.

## Run locally
1. Install Python 3.10+.
2. Create/activate a virtual environment (recommended).
3. Install dependencies: `pip install -r requirements.txt`
4. Open Jupyter and run every cell in `Assignment_2_End_to_End_ML_System.ipynb` from top to bottom.
5. Confirm `artifacts/titanic_model.joblib` and `artifacts/heart_disease_model.joblib` exist.
6. Run the app from this folder: `streamlit run app.py`

The first notebook run requires internet access to retrieve the datasets. The heart-disease system uses four inputs only: age, sex, chest-pain type (`cp`), and maximum heart rate achieved (`thalach`).

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and the `artifacts/` directory containing both trained model files.
2. Make sure the notebook has been run successfully so the artifacts exist. Do not commit a model file until it has been generated.
3. Sign in to Streamlit Community Cloud and deploy the repository, selecting `app.py` as the app entry point.
4. After deployment succeeds, copy the public app URL into the **Deployment link** placeholder at the top of the notebook and submit it as required.

A real live URL cannot be supplied until you deploy the app using your own hosting account.

## Important notes
- Results are calculated at runtime and can vary with dataset version/environment.
- This project is educational. The heart disease classifier is not a medical diagnostic tool.
- Follow your course's academic-integrity policy and ensure you understand the code before submission.
