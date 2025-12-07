import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = None
        self.best_model = None
        
    def load_data(self, train_path, test_path=None):

        self.train_df = pd.read_csv(train_path)
        print(f"Training data shape: {self.train_df.shape}")
        
        if test_path:
            self.test_df = pd.read_csv(test_path)
            print(f"Test data shape: {self.test_df.shape}")
        else:
            self.test_df = None
            
        return self
    
    def explore_data(self):

        print("\n=== Data Exploration ===")
        print(f"\nTraining Data Shape: {self.train_df.shape}")
        print(f"Columns: {list(self.train_df.columns)}")
        
        print(f"\nMissing Values:")
        missing = self.train_df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        
        print(f"\nTarget Distribution (status_gagal_bayar):")
        if 'status_gagal_bayar' in self.train_df.columns:
            target_dist = self.train_df['status_gagal_bayar'].value_counts()
            print(target_dist)
            print(f"Default Rate: {(target_dist[1] / len(self.train_df) * 100):.2f}%")
        elif 'target' in self.train_df.columns:
            target_dist = self.train_df['target'].value_counts()
            print(target_dist)
        
        # Categorical columns analysis
        categorical_cols = self.train_df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"\nCategorical Variables:")
            for col in categorical_cols[:5]:  # Show first 5
                print(f"  {col}: {self.train_df[col].nunique()} unique values")
                print(f"    Top values: {self.train_df[col].value_counts().head(3).to_dict()}")
        
        # Numerical columns summary
        numerical_cols = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            print(f"\nNumerical Variables Summary:")
            print(self.train_df[numerical_cols].describe())
        
        print(f"\nFirst 3 rows:")
        print(self.train_df.head(3))
        
    def preprocess_data(self, target_col='status_gagal_bayar', id_col='id_transaksi'):

        print("\n=== Preprocessing Data ===")
        
        # Separate features and target
        if target_col in self.train_df.columns:
            self.y_train = self.train_df[target_col]
            # Drop target and ID columns from features
            cols_to_drop = [target_col]
            if id_col in self.train_df.columns:
                cols_to_drop.append(id_col)
                self.train_id_col = id_col
            self.X_train = self.train_df.drop(columns=cols_to_drop)
        else:
            self.X_train = self.train_df.copy()
            self.y_train = None
            # Still check for ID column
            if id_col in self.train_df.columns:
                self.train_id_col = id_col
            else:
                self.train_id_col = None
            
        # Handle test data
        if self.test_df is not None:
            # Store ID column for test data
            if id_col in self.test_df.columns:
                self.test_id_col = id_col
                self.test_ids = self.test_df[id_col].copy()
                # Drop ID column from test features
                self.X_test = self.test_df.drop(columns=[id_col])
            else:
                self.test_id_col = None
                self.test_ids = None
                self.X_test = self.test_df.copy()
        else:
            self.X_test = None
            self.test_id_col = None
            self.test_ids = None
        
        # Identify date columns (will be processed in feature_engineering)
        date_cols = []
        for col in self.X_train.columns:
            if 'tanggal' in col.lower() or 'date' in col.lower():
                date_cols.append(col)
        
        # Identify categorical and numerical columns (excluding date columns)
        categorical_cols = [col for col in self.X_train.select_dtypes(include=['object', 'category']).columns 
                           if col not in date_cols]
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_cols:
            print(f"Date columns (will be processed in feature engineering): {date_cols}")
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # Handle missing values in numerical columns
        for col in numerical_cols:
            if self.X_train[col].isnull().sum() > 0:
                median_val = self.X_train[col].median()
                self.X_train[col].fillna(median_val, inplace=True)
                if self.X_test is not None:
                    self.X_test[col].fillna(median_val, inplace=True)
        
        # Handle missing values in categorical columns
        for col in categorical_cols:
            if self.X_train[col].isnull().sum() > 0:
                mode_val = self.X_train[col].mode()[0] if len(self.X_train[col].mode()) > 0 else 'Unknown'
                self.X_train[col].fillna(mode_val, inplace=True)
                if self.X_test is not None:
                    self.X_test[col].fillna(mode_val, inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit on combined data to handle unseen categories
            combined = pd.concat([self.X_train[col], self.X_test[col] if self.X_test is not None else pd.Series()])
            le.fit(combined.astype(str))
            
            self.X_train[col] = le.transform(self.X_train[col].astype(str))
            if self.X_test is not None:
                self.X_test[col] = le.transform(self.X_test[col].astype(str))
            
            self.label_encoders[col] = le
        
        # Store feature columns
        self.feature_columns = self.X_train.columns.tolist()
        
        print(f"\nPreprocessed training data shape: {self.X_train.shape}")
        if self.X_test is not None:
            print(f"Preprocessed test data shape: {self.X_test.shape}")
        
        return self
    
    def feature_engineering(self):

        print("\n=== Feature Engineering ===")
        
        for df_name, df in [('train', self.X_train), ('test', self.X_test)]:
            if df is None:
                continue
                
            if 'tanggal_pencairan' in df.columns:
                df['tanggal_pencairan'] = pd.to_datetime(df['tanggal_pencairan'], errors='coerce')
                
                df['tahun_pencairan'] = df['tanggal_pencairan'].dt.year
                df['bulan_pencairan'] = df['tanggal_pencairan'].dt.month
                df['hari_dalam_bulan'] = df['tanggal_pencairan'].dt.day
                df['hari_dalam_minggu'] = df['tanggal_pencairan'].dt.dayofweek
                df['kuartal'] = df['tanggal_pencairan'].dt.quarter
                
                df.drop(columns=['tanggal_pencairan'], inplace=True)
                print(f"  - Extracted date features from tanggal_pencairan ({df_name})")
        
        for df_name, df in [('train', self.X_train), ('test', self.X_test)]:
            if df is None:
                continue
                
            if 'jumlah_pinjaman' in df.columns and 'total_pengembalian' in df.columns:
                if 'bunga_pinjaman' not in df.columns and 'bunga' not in df.columns:
                    df['bunga_pinjaman'] = df['total_pengembalian'] - df['jumlah_pinjaman']
                elif 'bunga' in df.columns and 'bunga_pinjaman' not in df.columns:
                    df['bunga_pinjaman'] = df['bunga']
                
                if 'tingkat_bunga' not in df.columns:
                    df['tingkat_bunga'] = np.where(
                        df['jumlah_pinjaman'] > 0,
                        (df['bunga_pinjaman'] / df['jumlah_pinjaman']) * 100,
                        0
                    )
                
                df['rasio_pengembalian'] = np.where(
                    df['jumlah_pinjaman'] > 0,
                    df['total_pengembalian'] / df['jumlah_pinjaman'],
                    1
                )
                print(f"  - Created/verified financial ratios ({df_name})")
            
            if 'porsi_pengembalian_lender' in df.columns and 'total_pengembalian' in df.columns:
                df['rasio_porsi_lender'] = np.where(
                    df['total_pengembalian'] > 0,
                    df['porsi_pengembalian_lender'] / df['total_pengembalian'],
                    0
                )
            
            if 'jumlah_pinjaman' in df.columns and 'durasi_hari' in df.columns:
                df['pinjaman_per_hari'] = np.where(
                    df['durasi_hari'] > 0,
                    df['jumlah_pinjaman'] / df['durasi_hari'],
                    0
                )
                df['pengembalian_per_hari'] = np.where(
                    df['durasi_hari'] > 0,
                    df['total_pengembalian'] / df['durasi_hari'],
                    0
                )
            
            if 'jumlah_pinjaman' in df.columns:
                # Categorize loan amount (small, medium, large)
                df['kategori_pinjaman'] = pd.cut(
                    df['jumlah_pinjaman'],
                    bins=[0, 200000, 800000, float('inf')],
                    labels=[0, 1, 2]
                ).astype(int)
            
            if 'durasi_hari' in df.columns:
                df['kategori_durasi'] = pd.cut(
                    df['durasi_hari'],
                    bins=[0, 7, 30, float('inf')],
                    labels=[0, 1, 2]
                ).astype(int)
        
        if self.X_test is not None:
            train_cols = set(self.X_train.columns)
            test_cols = set(self.X_test.columns)
            
            for col in test_cols - train_cols:
                if pd.api.types.is_numeric_dtype(self.X_test[col]):
                    fill_value = self.X_test[col].median()
                    self.X_train[col] = fill_value
                else:
                    fill_value = self.X_test[col].mode()[0] if len(self.X_test[col].mode()) > 0 else 0
                    self.X_train[col] = fill_value
                print(f"  - Added missing column '{col}' to training data")
            
            for col in train_cols - test_cols:
                if pd.api.types.is_numeric_dtype(self.X_train[col]):
                    fill_value = self.X_train[col].median()
                    self.X_test[col] = fill_value
                else:
                    fill_value = self.X_train[col].mode()[0] if len(self.X_train[col].mode()) > 0 else 0
                    self.X_test[col] = fill_value
                print(f"  - Added missing column '{col}' to test data")
            
            self.X_test = self.X_test[self.X_train.columns]
        
        print("Feature engineering completed.")
        print(f"Training features: {self.X_train.shape[1]} columns")
        if self.X_test is not None:
            print(f"Test features: {self.X_test.shape[1]} columns")
            if set(self.X_train.columns) == set(self.X_test.columns):
                print("✓ Train and test have matching columns")
            else:
                print("⚠ Warning: Train and test columns don't match!")
        
        return self
    
    def train_models(self, use_cv=True, cv_folds=5):
        """
        Train multiple models and select the best one.
        
        Args:
            use_cv: Whether to use cross-validation
            cv_folds: Number of CV folds
        """
        print("\n=== Training Models ===")
        
        if self.y_train is None:
            print("No target variable found. Skipping training.")
            return self
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        self.scalers['main'] = scaler
        
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            if use_cv:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train_scaled, self.y_train, 
                                        cv=cv, scoring='f1_macro')
                mean_score = scores.mean()
                std_score = scores.std()
                print(f"CV Macro F1-Score: {mean_score:.4f} (+/- {std_score:.4f})")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
            else:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_scaled, self.y_train, test_size=0.2, 
                    random_state=42, stratify=self.y_train
                )
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='macro')
                print(f"Validation Macro F1-Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
            
            model.fit(X_train_scaled, self.y_train)
            self.models[name] = model
        
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name} with Macro F1-Score: {best_score:.4f}")
        
        return self
    
    def evaluate_macro_f1(self, y_true, y_pred):
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"\nMacro F1-Score: {macro_f1:.4f}")
        
        if len(np.unique(y_true)) == 2:
            f1_class_0 = f1_score(y_true, y_pred, average=None, labels=[0])[0]
            f1_class_1 = f1_score(y_true, y_pred, average=None, labels=[1])[0]
            print(f"F1-Score Class 0: {f1_class_0:.4f}")
            print(f"F1-Score Class 1: {f1_class_1:.4f}")
            print(f"Macro F1 = (F1_class_0 + F1_class_1) / 2 = ({f1_class_0:.4f} + {f1_class_1:.4f}) / 2 = {macro_f1:.4f}")
        
        return macro_f1
    
    def predict(self, return_proba=False):
        if self.best_model is None:
            print("No model trained yet. Please train models first.")
            return None
        
        if self.X_test is None:
            print("No test data available.")
            return None
        
        print("\n=== Making Predictions ===")
        
        # Scale test data
        X_test_scaled = self.scalers['main'].transform(self.X_test)
        
        if return_proba:
            predictions = self.best_model.predict_proba(X_test_scaled)[:, 1]
            print("Returning probability predictions.")
        else:
            predictions = self.best_model.predict(X_test_scaled)
            print("Returning class predictions (0 or 1) for Macro F1-Score evaluation.")
        
        print(f"Predictions shape: {predictions.shape}")
        return predictions
    
    def save_predictions(self, predictions, output_path='submission.csv', id_col='id_transaksi'):
        if self.X_test is None:
            print("No test data available.")
            return
        
        # Use stored test IDs if available, otherwise try to get from X_test
        if self.test_ids is not None:
            ids = self.test_ids
        elif id_col in self.X_test.columns:
            ids = self.X_test[id_col]
        else:
            # Fallback: generate sequential IDs
            ids = range(len(predictions))
            print(f"Warning: {id_col} not found. Using sequential IDs.")
        
        # Ensure predictions are integers (0 or 1)
        predictions = predictions.astype(int)
        
        submission_df = pd.DataFrame({
            'id_transaksi': ids,
            'status_gagal_bayar': predictions
        })
        
        submission_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        print(f"Submission format: id_transaksi, status_gagal_bayar")
        print(f"Shape: {submission_df.shape}")
        print(f"\nFirst few rows:")
        print(submission_df.head())
        return submission_df


def main():
    import os
    
    model = CreditRiskModel()
    
    train_files = ['train.csv', 'train_data.csv', 'training.csv']
    test_files = ['test.csv', 'test_data.csv', 'testing.csv']
    
    train_path = None
    test_path = None
    
    for file in train_files:
        if os.path.exists(file):
            train_path = file
            break
    
    for file in test_files:
        if os.path.exists(file):
            test_path = file
            break
    
    if train_path is None:
        print("=" * 60)
        print("ERROR: Training data file not found!")
        print("=" * 60)
        print("\nPlease:")
        print("1. Download the data using: python download_data.py")
        print("   OR")
        print("2. Run: kaggle competitions download -c adikara-2025-indonesia-kredit-macet")
        print("3. Extract the ZIP file")
        print("4. Make sure train.csv is in the current directory")
        print("\nAlternatively, specify the path manually:")
        print("  model.load_data('path/to/train.csv', 'path/to/test.csv')")
        return
    
    print("=" * 60)
    print("Credit Risk Prediction Pipeline")
    print("=" * 60)
    
    if test_path:
        model.load_data(train_path, test_path)
    else:
        print("Warning: Test file not found. Will only train the model.")
        model.load_data(train_path)
    
    model.explore_data()
    model.preprocess_data(target_col='status_gagal_bayar', id_col='id_transaksi')
    model.feature_engineering()
    model.train_models(use_cv=True, cv_folds=5)
    
    if test_path:
        predictions = model.predict(return_proba=False)
        model.save_predictions(predictions, 'submission.csv', id_col='id_transaksi')
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("Submission file saved as: submission.csv")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Model training completed!")
        print("To make predictions, provide test.csv and run predict()")
        print("=" * 60)


if __name__ == "__main__":
    main()

