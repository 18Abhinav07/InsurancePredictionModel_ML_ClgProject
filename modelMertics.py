import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import joblib
import seaborn as sns

class InsuranceModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.metrics = {}
        self.feature_importance = None
        self.best_model = None
        self.report_text = ""

    def prepare_data(self, df):
        # Create and fit label encoders for categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        return df

    def train(self, data_path):
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df = self.prepare_data(df)

        # Split features and target
        X = df.drop('charges', axis=1)
        y = df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train and tune Random Forest
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='r2', n_jobs=-1)
        rf_grid.fit(X_train_scaled, y_train)
        self.models['RandomForest'] = rf_grid.best_estimator_

        # Train and tune Gradient Boosting
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='r2', n_jobs=-1)
        gb_grid.fit(X_train_scaled, y_train)
        self.models['GradientBoosting'] = gb_grid.best_estimator_

        # Evaluate models
        self.evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
        self.select_best_model()
        self.generate_visualizations(X, X_test_scaled, y_test)
        self.save_artifacts()

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            self.metrics[name] = {
                'train_r2': r2_score(y_train, train_pred),
                'test_r2': r2_score(y_test, test_pred),
                'train_mse': mean_squared_error(y_train, train_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'test_mae': mean_absolute_error(y_test, test_pred)
            }

    def select_best_model(self):
        # Select the model with the highest test R2 score
        self.best_model = max(self.models, key=lambda name: self.metrics[name]['test_r2'])
        self.report_text += f"\nBest Model: {self.best_model}\nMetrics: {self.metrics[self.best_model]}\n"

    def generate_visualizations(self, X, X_test_scaled, y_test):
        plt.figure(figsize=(10, 6))
        for name, model in self.models.items():
            test_pred = model.predict(X_test_scaled)
            sns.scatterplot(x=y_test, y=test_pred, label=name)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
        plt.xlabel('Actual Charges')
        plt.ylabel('Predicted Charges')
        plt.title('Actual vs Predicted Charges')
        plt.legend()
        plt.savefig('static/actual_vs_predicted.png')

        # Feature importance for Random Forest
        rf_feature_importance = pd.Series(
            self.models['RandomForest'].feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        rf_feature_importance.plot(kind='bar')
        plt.title('Random Forest Feature Importance')
        plt.ylabel('Importance Score')
        plt.xlabel('Features')
        plt.tight_layout()
        plt.savefig('static/feature_importance.png')

        self.feature_importance = rf_feature_importance.to_dict()

    def save_artifacts(self):
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name.lower()}_model.joblib')

        # Save preprocessors and metrics
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.encoders, 'models/encoders.joblib')
        with open('static/metrics.json', 'w') as f:
            json.dump(self.metrics, f)
        with open('static/feature_importance.json', 'w') as f:
            json.dump(self.feature_importance, f)

        # Save the report
        with open('static/report.txt', 'w') as f:
            f.write(self.report_text)

if __name__ == "__main__":
    model = InsuranceModel()
    model.train('insurance.csv')
