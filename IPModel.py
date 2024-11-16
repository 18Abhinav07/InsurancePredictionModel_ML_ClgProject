import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib

class InsuranceModel:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = None
        self.metrics = {}

    def prepare_data(self, df):
        # Create and fit label encoders
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df

    def train(self, data_path):
        # Load and prepare data
        df = pd.read_csv(data_path)
        df = self.prepare_data(df)
        
        # Split features and target
        X = df.drop('charges', axis=1)
        y = df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                 cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        self.rf_model = grid_search.best_estimator_
        
        # Gradient Boosting model
        self.gb_model = GradientBoostingRegressor(n_estimators=200, 
                                                 learning_rate=0.1,
                                                 max_depth=5,
                                                 random_state=42)
        self.gb_model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        self.calculate_metrics(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Generate and save visualizations
        self.generate_visualizations(X, X_test_scaled, y_test)
        
        # Save models and preprocessors
        self.save_artifacts()

    def calculate_metrics(self, X_train_scaled, X_test_scaled, y_train, y_test):
        # Random Forest predictions
        rf_train_pred = self.rf_model.predict(X_train_scaled)
        rf_test_pred = self.rf_model.predict(X_test_scaled)
        
        # Gradient Boosting predictions
        gb_train_pred = self.gb_model.predict(X_train_scaled)
        gb_test_pred = self.gb_model.predict(X_test_scaled)
        
        # Store metrics
        self.metrics = {
            'random_forest': {
                'train_r2': r2_score(y_train, rf_train_pred),
                'test_r2': r2_score(y_test, rf_test_pred),
                'train_mse': mean_squared_error(y_train, rf_train_pred),
                'test_mse': mean_squared_error(y_test, rf_test_pred),
                'train_mae': mean_absolute_error(y_train, rf_train_pred),
                'test_mae': mean_absolute_error(y_test, rf_test_pred)
            },
            'gradient_boosting': {
                'train_r2': r2_score(y_train, gb_train_pred),
                'test_r2': r2_score(y_test, gb_test_pred),
                'train_mse': mean_squared_error(y_train, gb_train_pred),
                'test_mse': mean_squared_error(y_test, gb_test_pred),
                'train_mae': mean_absolute_error(y_train, gb_train_pred),
                'test_mae': mean_absolute_error(y_test, gb_test_pred)
            }
        }

    def generate_visualizations(self, X, X_test_scaled, y_test):
        # Feature importance
        self.feature_importance = pd.Series(
            self.rf_model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)

        # Save feature importance to JSON
        with open('static/feature_importance.json', 'w') as f:
            json.dump(
                {str(k): float(v) for k, v in self.feature_importance.items()},
                f
            )

    def save_artifacts(self):
        # Save models
        joblib.dump(self.rf_model, 'models/rf_model.joblib')
        joblib.dump(self.gb_model, 'models/gb_model.joblib')
        
        # Save preprocessors
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.encoders, 'models/encoders.joblib')
        
        # Save metrics
        with open('static/metrics.json', 'w') as f:
            json.dump(self.metrics, f)

if __name__ == "__main__":
    model = InsuranceModel()
    model.train('insurance.csv')
