# ADIKARA 2025 - Credit Risk Prediction

Kaggle competition for predicting credit default risk in microfinance and SME sector in Indonesia.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure you have the data files
- `train.csv` - Training data with target column `status_gagal_bayar`
- `test.csv` - Test data for predictions

### 3. Run the Model

```bash
python model.py
```

The script will automatically:
- Find `train.csv` and `test.csv` in the current directory
- Explore the data
- Preprocess and engineer features
- Train multiple models with cross-validation
- Select the best model based on Macro F1-Score
- Generate `submission.csv` for Kaggle submission

## Evaluation Metric

- **Macro F1-Score**: The competition uses Macro-averaged F1-Score
- Formula: `Macro F1 = (F1_class_0 + F1_class_1) / 2`
- Implementation: `f1_score(y_true, y_pred, average='macro')`

## Submission Format

The submission file (`submission.csv`) will have:
- Header: `id_transaksi,status_gagal_bayar`
- Predictions: Binary values (0 or 1) for `status_gagal_bayar`

## Files

- `model.py` - Main model pipeline with complete ML workflow
- `requirements.txt` - Python dependencies
- `sample_submissions.csv` - Sample submission format reference

## Customization

You can customize the model by:
- Modifying `feature_engineering()` to add domain-specific features
- Adjusting model hyperparameters in `train_models()`
- Adding more models to the ensemble (XGBoost, LightGBM, CatBoost, etc.)

## Data Columns

Based on the competition, the data includes:
- **Categorical**: `provinsi`, `jenis_pinjaman`, `status_peminjam`, `sektor_usaha`, `pendidikan`, `jenis_jaminan`
- **Numerical**: `jumlah_pinjaman`, `total_pengembalian`, `durasi_hari`, `porsi_pengembalian_lender`
- **Target**: `status_gagal_bayar` (0 = no default, 1 = default)
