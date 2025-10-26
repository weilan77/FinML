import pandas as pd
import datetime
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from numpy import log, polyfit, sqrt, std, subtract
from statsmodels.tsa.stattools import adfuller

def fit(self, df, validate_ratio=0.2):
        """
        训练模型（包含状态语义映射）
        """
        print("🚀 开始训练 ConsistentMarketPredictor...")
        
        X_train, y_train, feature_names = self.prepare_training_data(df)
        
        # 移除未知状态样本
        valid_mask = y_train != -1
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        if len(X_train) == 0:
            raise ValueError("没有有效的训练样本")
        
        split_idx = int(len(X_train) * (1 - validate_ratio))
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        
        print("步骤5: 训练状态预测器...")
        self.state_predictor = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.state_predictor.fit(
            X_tr_scaled, y_tr,
            eval_set=[(X_val_scaled, y_val)],
            eval_metric='multi_error',
            early_stopping_rounds=50,
            verbose=False
        )
        
        # 验证性能
        val_pred = self.state_predictor.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, val_pred)
        
        print(f"✅ 训练完成!")
        print(f"验证集准确率: {accuracy:.3f}")
        print(f"状态分布: {pd.Series(y_val).value_counts().to_dict()}")
        
        self.is_trained = True
        return accuracy
    
def rolling_adf(series, window=100):
    def adf_calc(x):
        result = adfuller(x)
        return result[0]  # 返回ADF统计量

    return series.rolling(window).apply(adf_calc, raw=True)

print("adf calc finsh")
print("End")

print("sig1")

