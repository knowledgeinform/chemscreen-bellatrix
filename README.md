# AI-H2: Machine Learning for High-Entropy Materials Discovery

This repository contains a pipeline for discovering high-entropy materials using machine learning, initialized with DFT data (EFA/DEED) and refined through active learning with experimental feedback.

## Project Structure

The codebase is organized into the following functional modules:

### 1. Data Preprocessing (`data_preprocessing/`)
- `chem_pool/`: Tools for generating all possible chemical compositions (e.g., 5-cation combinations).
- `efa_deed_calc/`: Scripts for calculating EFA (Entropy Forming Ability) and DEED (Descriptor for Energy of Disorder) using AFLOW and DFT data.

### 2. Model Training and Evaluation (`Training/`)
- `RF/`: Random Forest model training, cross-validation, and SHAP analysis.
- `analysis/`: Post-processing visualization tools (e.g., parity plots, element distribution analysis).
- `model_update/`: **Active Learning** module. Contains `update_model_with_feedback.py` which adjusts the model based on experimental results (`exp_truth`).
  - **Reward/Penalty Logic**: Successful syntheses are rewarded (coefficient < 1.0 for EFA/DEED, as lower values are often better), and failures are penalized (coefficient > 1.0).
  - **Retraining**: Refines the model using the rewarded experimental data.

### 3. Model Application (`application/`)
- `run_prediction_and_recommendation.py`: The main entry point for applying a trained model to new compositions.
  - **Deduplication**: Ensures no duplicate compositions are processed.
  - **Prediction**: Generates ML predictions for the entire chemical pool.
  - **Recommendation**: Identifies the top N candidates with the highest predicted EFA/DEED for the next experimental cycle.

### 4. Archive (`archive/`)
- Contains legacy code, temporary scripts, and unused data files to keep the main workspace clean.

### 5. Data and Output
- `Data/`: Contains raw and processed datasets, including chemical pools and experimental feedback.
- `Output/`: (Ignored by Git) Stores model checkpoints, predictions, and visualization results.

## Workflow

1. **Generate Pool**: Use `data_preprocessing/chem_pool/comb_gen_all.py` to create a list of candidate compositions.
2. **Initial Training**: Train a base model in `Training/RF/` using available DFT data.
3. **Predict & Recommend**: Run `application/run_prediction_and_recommendation.py` to get the first batch of experimental candidates.
4. **Experimental Feedback**: After experiments, update the `Data/exp_truth/` files.
5. **Update Model**: Run `Training/model_update/update_model_with_feedback.py` to refine the model based on the new experimental data.
6. **Iterate**: Repeat steps 3-5 to explore the chemical space efficiently.

## Installation & Requirements

Ensure you have the following dependencies installed:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `pickle`

The project assumes a specific directory structure for `mat_encod` (material encoding utilities), which should be present in the root.
