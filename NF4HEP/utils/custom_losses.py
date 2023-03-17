import tensorflow as tf # type: ignore
from tensorflow.python.keras import backend as K # type: ignore
from tensorflow.keras.losses import Loss # type: ignore
from typing import Any

class minus_y_pred(Loss):
    """
    Minus y_pred loss
    """
    def __init__(self) -> None:
        super().__init__(name="minus_y_pred")
        
    def call(self,
             y_true: Any, 
             y_pred: Any) -> Any:
        return -y_pred

class mean_error(Loss):
    """
    Minus y_pred loss
    """
    def __init__(self) -> None:
        super().__init__(name="mean_error")
        
    def call(self,
             y_true: Any, 
             y_pred: Any) -> Any:
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        ME_model = K.mean(y_true-y_pred)
        return K.abs(ME_model)
    
class mean_percentage_error(Loss):
    """
    Minus y_pred loss
    """
    def __init__(self) -> None:
        super().__init__(name="mean_percentage_error")
        
    def call(self,
             y_true: Any, 
             y_pred: Any) -> Any:
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        MPE_model = K.mean((y_true-y_pred)/(K.sign(y_true)*K.clip(K.abs(y_true),
                                                                  K.epsilon(),
                                                                  None)))
        return 100. * K.abs(MPE_model)
    
class R2_metric(Loss):
    """
    Minus y_pred loss
    """
    def __init__(self) -> None:
        super().__init__(name="R2_metric")
        
    def call(self,
             y_true: Any, 
             y_pred: Any) -> Any:
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        MSE_model =  K.sum(K.square( y_true-y_pred )) 
        MSE_baseline = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return (1 - MSE_model/(MSE_baseline + K.epsilon()))

def losses():
    return {"minus_y_pred": minus_y_pred,
            "mean_error": mean_error,
            "mean_percentage_error": mean_percentage_error,
            "R2_metric": R2_metric}

def metric_name_abbreviate(metric_name):
    name_dict = {"accuracy": "acc",
                 "minus_y_pred": "myp",
                 "mean_error": "me",
                 "mean_percentage_error": "mpe", 
                 "mean_squared_error": "mse",
                 "mean_absolute_error": "mae", 
                 "mean_absolute_percentage_error": "mape", 
                 "mean_squared_logarithmic_error": "msle"}
    for key in name_dict:
        metric_name = metric_name.replace(key, name_dict[key])
    return metric_name

def metric_name_unabbreviate(metric_name):
    name_dict = {"acc": "accuracy", 
                 "myp": "minus_y_pred",
                 "me": "mean_error", 
                 "mpe": "mean_percentage_error", 
                 "mse": "mean_squared_error",
                 "mae": "mean_absolute_error", 
                 "mape": "mean_absolute_percentage_error", 
                 "msle": "mean_squared_logarithmic_error"}
    for key in name_dict:
        metric_name = metric_name.replace(key, name_dict[key])
    return metric_name
