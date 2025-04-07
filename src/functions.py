from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, BayesianRidge, LassoCV, ElasticNetCV, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel, RFE
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FunctionTransformer, Pipeline
import pandas as pd
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

# models = {
#         'ElasticNet': ElasticNet(random_state=42),
#         'SVR': SVR(),
#         'BayesianRidge': BayesianRidge()
#     }

def getFeats(df):
    # return df.drop(columns=["Unnamed: 0", "BMI"])
    return df.drop(columns=["BMI"])

def getTarget(df):
    return df["BMI"]

def trainBaseModel(dfeat, dtarg, vfeat, vtarg, model_type='ElasticNet'):
    if model_type == 'ElasticNet':
        model = ElasticNet(random_state=42)
    elif model_type == 'SVR':
        model = SVR()
    elif model_type == 'BayesianRidge':
        model = BayesianRidge()
    else:
        raise ValueError("Invalid model type")
    
    # Train model
    model.fit(dfeat, dtarg)
    
    # Predict on evaluation set
    predictions = model.predict(vfeat)
    
    # Evaluate metrics
    mse = mean_squared_error(vtarg, predictions)
    mae = mean_absolute_error(vtarg, predictions)
    r2 = r2_score(vtarg, predictions)
    rmse = np.sqrt(mse)
    
    return model, {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


# def load_data(dev_path, eval_path):
#     # Load and merge development/evaluation datasets"
#     dev_df = pd.read_csv(dev_path)
#     eval_df = pd.read_csv(eval_path)
#     return pd.concat([dev_df, eval_df], axis=0)

def selectFeats(dev, alphas):
    selector = LassoCV(cv=5, random_state=42, max_iter=10000000, n_jobs=-1, alphas=alphas)
    selector.fit(getFeats(dev), getTarget(dev))
    sfm = SelectFromModel(selector, prefit=True)
    return getFeats(dev).columns[sfm.get_support()]

def selectFeatsLasso(dev):
    lasso = Lasso(alpha=0.1, max_iter=100000)
    lasso.fit(getFeats(dev), getTarget(dev))
    selFeat = getFeats(dev).columns[lasso.coef_ != 0]
    devSel = getFeats(dev)[selFeat]
    lasso2 = Lasso(alpha=0.01, max_iter=100000)
    lasso2.fit(devSel, getTarget(dev))
    selFeat2 = devSel.columns[lasso2.coef_ != 0]
    return selFeat2

def selectFeatRFE(dev, featN):
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=featN)
    rfe.fit(getFeats(dev), getTarget(dev))

    selFeat = getFeats(dev).columns[rfe.support_]
    return selFeat

def selectFeatBest(dev, ku, f):
    from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    devScaled = scaler.fit_transform(getFeats(dev))
    sel = SelectKBest(score_func=f, k=ku)
    bfs = sel.fit_transform(devScaled, getTarget(dev))
    return getFeats(dev).columns[sel.get_support()]

def selectFeatsENCV(dev, l1):
    selector = ElasticNetCV(cv=5, random_state=42, l1_ratio=l1, max_iter=100000, n_jobs=-1)
    selector.fit(getFeats(dev), getTarget(dev))
    # sfm = SelectFromModel(selector, prefit=True)
    # return getFeats(dev).columns[sfm.get_support()]
    print(selector.l1_ratio_)
    return [selector.feature_names_in_[i] for i in range(len(selector.coef_)) if selector.coef_[i] != 0]


def selectFeatsXGB(dev):
    xgbModel = XGBRegressor()
    xgbModel.fit(getFeats(dev), getTarget(dev))
    sRank = pd.Series(xgbModel.feature_importances_, index=getFeats(dev).columns).nlargest(76)

    # https://github.com/scikit-learn-contrib/stability-selection remake
    bs = 100
    th = 0.8
    featCount = np.zeros(getFeats(dev)[sRank.index].shape[1])
    coefMatrix = np.zeros((bs, getFeats(dev)[sRank.index].shape[1]))

    for i in range(bs):
        featResampled, tarResampled = resample(getFeats(dev)[sRank.index], getTarget(dev), random_state=1)
        lassoModel = LassoCV(cv=5, max_iter=100000)
        lassoModel.fit(featResampled, tarResampled)

        coefMatrix[i] = lassoModel.coef_
        featCount += (lassoModel.coef_ != 0).astype(int)
    stability = featCount/bs
    selFeat = np.where(stability > th)[0]
    
    return lassoModel.feature_names_in_[selFeat]


def tuneModel(data, selFeats, model_type='ElasticNet'):
    feats = getFeats(data)[selFeats]
    # feats = getFeats(data)
    # print(feats.head())
    targt = getTarget(data)
    # Hyperparameter tuning with GridSearchCV
    param_grids = {
        'ElasticNet': {'model__alpha': [0.1, 1, 10], 'model__l1_ratio': [0.2, 0.5, 0.8]},
        'SVR': {'model__C': [0.1, 1, 10], 'model__epsilon': [0.01, 0.1]},
        'BayesianRidge': {'model__alpha_1': [1e-6, 1e-5], 'model__lambda_1': [1e-6, 1e-5]}
    }
    # cv=5, random_state=42, max_iter=10000000, n_jobs=-1, alphas=alphas
    
    if model_type == 'ElasticNet':
        model = ElasticNet(random_state=42)
    elif model_type == 'SVR':
        model = SVR()
    elif model_type == 'BayesianRidge':
        model = BayesianRidge()
    else:
        raise ValueError("Invalid model type")

    # pipeline = Pipeline(steps=[
    #     ('scaler', StandardScaler()),
    #     ('sfm', SelectFromModel( estimator=LassoCV(cv=5, random_state=42, max_iter=10000000, n_jobs=-1, alphas=[0.2]) )),
    #     ('model', model)
    # ])
    # print(pipeline.get_params().keys())
    pipeline = Pipeline([
        ('model', model)
    ])
    
    grid = GridSearchCV(pipeline, param_grids[model_type], cv=5, scoring='neg_root_mean_squared_error')
    grid.fit(feats, targt)
    return grid.best_estimator_

def evalModel(model, X_test, y_test):
    # Calculate evaluation metrics
    preds = model.predict(X_test)


    # mse = mean_squared_error(vtarg, predictions)
    # mae = mean_absolute_error(vtarg, predictions)
    # r2 = r2_score(vtarg, predictions)
    # rmse = np.sqrt(mse)
    
    # return model, {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    return {
        'MSE': mean_squared_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE': mean_absolute_error(y_test, preds),
        'R2': r2_score(y_test, preds)
    }


def modelStatsDiff(dict_a, dict_b):
    # Compute absolute and percentage differences between two model metric dictionaries.
    comparisons = {}
    
    # Get all unique model names from both dictionaries
    all_models = set(dict_a.keys()).union(dict_b.keys())
    
    for model in all_models:
        if model in dict_a and model in dict_b:
            comparisons[model] = {}
            for metric in dict_a[model]:
                val_a = dict_a[model][metric]
                val_b = dict_b[model][metric]
                
                abs_diff = val_b - val_a
                avg = (val_a + val_b) / 2
                pct_diff = (abs_diff / avg) * 100 if avg != 0 else 0
                
                comparisons[model][metric] = {
                    'Absolute': abs_diff,
                    'Percentage': pct_diff
                }
        else:
            print(f"Warning: {model} missing in one dictionary")
            
    return comparisons

def printModelDiff(base, next):
    for md, mt in modelStatsDiff(base, next).items():
        print(f"\n## {md} Comparison")
        print(f"{'Metric':<10} {'Version 1':<12} {'Version 2':<12} {'Absolute Δ':<12} {'% Δ':<8}")
        print("-" * 56)
        for metric, values in mt.items():
            v1 = base[md][metric]
            v2 = next[md][metric]
            print(f"{metric:<10} {v1:<12.4f} {v2:<12.4f} {values['Absolute']:<12.4f} {values['Percentage']:<8.1f}")

def plotModelDiff(base, next, curLabel):
    # Plots the absolute and percentage differences for model metrics.
    for model, metrics in modelStatsDiff(base, next).items():
        metrics_list = list(metrics.keys())
        # abs_diffs = [metrics[metric]['Absolute'] for metric in metrics_list]
        pct_diffs = [metrics[metric]['Percentage'] for metric in metrics_list]

        # Bar plot for absolute differences
        x = np.arange(len(metrics_list))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 6))

        rects1 = ax1.bar(x - width/2, [base[model][metric] for metric in metrics_list], width, label='Baseline', color='teal')
        rects2 = ax1.bar(x + width/2, [next[model][metric] for metric in metrics_list], width, label=curLabel, color='orange')

        # Add labels, title, and legend
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title(f'{model} Metric Comparisons')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_list)
        ax1.legend()

        # Annotate bars
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax1.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # Plot percentage differences
        if any(pct_diffs):
            fig, ax2 = plt.subplots(figsize=(10, 6))
            ax2.bar(metrics_list, pct_diffs, color='orange', alpha=0.7)

            # Add labels, title, and legend
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Percentage Difference (%)')
            ax2.set_title(f'{model} Percentage Differences')

            # Annotate bars
            for i, v in enumerate(pct_diffs):
                if v != 0:
                    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

def labelEncodeSex(column):
    le = LabelEncoder()
    return le.fit_transform(column)

def modelToPipe(model):
    # numtransf = StandardScaler()
    # categtransf = OneHotEncoder()
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numtransf, [col for col in numerical_features if col != "sex"]),  # Scale numerical columns
    #         ('cat', categtransf, ["sex"])  # Encode categorical column
    #     ]
    # )
    pipeline = Pipeline(steps=[
        ('label_encoder', FunctionTransformer(lambda x: labelEncodeSex(x['sex']), validate=False)),
        ('scaler', StandardScaler()),
        ('sfm', SelectFromModel( estimator=LassoCV(cv=5, random_state=42, max_iter=10000000, n_jobs=-1, alphas=[0.2]) )),
        ('model', model)
    ])
    # pipeline = Pipeline(steps=[
    #     ('scaler', StandardScaler()),
    #     ('sfm', SelectFromModel( estimator=LassoCV(cv=5, random_state=42, max_iter=10000000, n_jobs=-1, alphas=[0.2]) )),
    #     ('model', model)
    # ])
    return pipeline