import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
import joblib
import seaborn as sns
from scipy import stats

class InsuranceModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.metrics = {}
        self.feature_importance = None
        self.best_model = None
        self.report_text = ""
        self.tuning_results = {}
        
    def prepare_data(self, df):
        # Create and fit label encoders for categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
            
        # Add interaction terms
        df['bmi_age'] = df['bmi'] * df['age']
        df['smoker_age'] = df['smoker'] * df['age']
        df['bmi_smoker'] = df['bmi'] * df['smoker']
        
        return df

    def plot_data_distributions(self, df):
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(df.columns, 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[col], kde=True)
            plt.title(f'{col} Distribution')
        plt.tight_layout()
        plt.savefig('static/distributions.png')
        plt.close()

    def plot_correlation_matrix(self, df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('static/correlation_matrix.png')
        plt.close()

    def plot_learning_curves(self, X, y, model_name, model):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='r2'
        )
        
        plt.figure(figsize=(10, 6))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        plt.xlabel('Training Examples')
        plt.ylabel('R² Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(f'static/learning_curve_{model_name.lower()}.png')
        plt.close()

    def plot_residuals(self, y_true, y_pred, model_name):
        residuals = y_true - y_pred
        plt.figure(figsize=(12, 4))
        
        # Residuals vs Predicted
        plt.subplot(121)
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted')
        
        # Residual Distribution
        plt.subplot(122)
        sns.histplot(residuals, kde=True)
        plt.title('Residual Distribution')
        
        plt.suptitle(f'{model_name} - Residual Analysis')
        plt.tight_layout()
        plt.savefig(f'static/residuals_{model_name.lower()}.png')
        plt.close()

    def train(self, data_path):
        # Load and preprocess data
        df = pd.read_csv(data_path)
        self.plot_data_distributions(df)
        
        df = self.prepare_data(df)
        self.plot_correlation_matrix(df)

        # Split features and target
        X = df.drop('charges', axis=1)
        y = df['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Enhanced Random Forest parameters
        rf_params = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Enhanced Gradient Boosting parameters
        gb_params = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }

        # Train models with enhanced parameters
        for name, (model, params) in {
            'RandomForest': (RandomForestRegressor(random_state=42), rf_params),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), gb_params)
        }.items():
            # Perform grid search
            grid = GridSearchCV(model, params, cv=5, scoring='r2', n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            
            # Store results
            self.models[name] = grid.best_estimator_
            self.tuning_results[name] = {
                'best_params': grid.best_params_,
                'cv_results': pd.DataFrame(grid.cv_results_)
            }
            
            # Generate learning curves
            self.plot_learning_curves(X_train_scaled, y_train, name, grid.best_estimator_)

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
            
            # Plot residuals
            self.plot_residuals(y_test, test_pred, name)

    def plot_hyperparameter_effects(self):
        for model_name, results in self.tuning_results.items():
            cv_results = results['cv_results']
            best_params = results['best_params']
            
            plt.figure(figsize=(15, 10))
            for i, param in enumerate(best_params.keys(), 1):
                plt.subplot(3, 2, i)
                param_values = cv_results[f'param_{param}'].astype(str)
                mean_scores = cv_results['mean_test_score']
                std_scores = cv_results['std_test_score']
                
                # Group by parameter value and calculate mean score
                scores_by_param = pd.DataFrame({
                    'value': param_values,
                    'score': mean_scores
                }).groupby('value')['score'].mean()
                
                plt.plot(range(len(scores_by_param)), scores_by_param.values, marker='o')
                plt.xticks(range(len(scores_by_param)), scores_by_param.index, rotation=45)
                plt.xlabel(param)
                plt.ylabel('Mean R² Score')
                plt.title(f'Effect of {param}')
            
            plt.suptitle(f'{model_name} - Hyperparameter Effects')
            plt.tight_layout()
            plt.savefig(f'static/hyperparameter_effects_{model_name.lower()}.png')
            plt.close()

    def generate_visualizations(self, X, X_test_scaled, y_test):
        # Original actual vs predicted plot
        plt.figure(figsize=(10, 6))
        for name, model in self.models.items():
            test_pred = model.predict(X_test_scaled)
            sns.scatterplot(x=y_test, y=test_pred, label=name, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
        plt.xlabel('Actual Charges')
        plt.ylabel('Predicted Charges')
        plt.title('Actual vs Predicted Charges')
        plt.legend()
        plt.savefig('static/actual_vs_predicted.png')
        plt.close()

        # Feature importance plots
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                
                plt.figure(figsize=(12, 6))
                importance.plot(kind='bar')
                plt.title(f'{name} - Feature Importance')
                plt.ylabel('Importance Score')
                plt.xlabel('Features')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'static/feature_importance_{name.lower()}.png')
                plt.close()

        # Plot hyperparameter effects
        self.plot_hyperparameter_effects()

    def save_artifacts(self):
        # Save models and preprocessors
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name.lower()}_model.joblib')
        
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.encoders, 'models/encoders.joblib')
        
        # Save metrics and results
        with open('static/metrics.json', 'w') as f:
            json.dump(self.metrics, f)
        
        # Save hyperparameter tuning results
        tuning_results_json = {
            name: {
                'best_params': results['best_params'],
                'cv_results': results['cv_results'].to_dict()
            }
            for name, results in self.tuning_results.items()
        }
        with open('static/tuning_results.json', 'w') as f:
            json.dump(tuning_results_json, f)
        
        # Generate comprehensive report
        self.generate_report()
        
    def generate_report(self):
        report = ["# Insurance Price Prediction Model Report\n"]
        
        # Model Performance Summary
        report.append("## Model Performance Summary")
        for name, metrics in self.metrics.items():
            report.append(f"\n### {name}")
            report.append(f"- Test R² Score: {metrics['test_r2']:.4f}")
            report.append(f"- Test MSE: {metrics['test_mse']:.2f}")
            report.append(f"- Test MAE: {metrics['test_mae']:.2f}")
        
        # Best Model Details
        report.append("\n## Best Model Details")
        report.append(f"Best performing model: {self.best_model}")
        best_params = self.tuning_results[self.best_model]['best_params']
        report.append("\nBest hyperparameters:")
        for param, value in best_params.items():
            report.append(f"- {param}: {value}")
        
        # Save report
        self.report_text = '\n'.join(report)
        with open('static/report.txt', 'w') as f:
            f.write(self.report_text)

if __name__ == "__main__":
    model = InsuranceModel()
    model.train('insurance.csv')