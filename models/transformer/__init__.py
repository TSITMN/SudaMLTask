from .transformer import Transformer
from .conv_transformer import Transformer as ConvTransformer
from .conv_periodpos_transformer import ConvTransformer as ConvPeriodPosTransformer
from .periodcpos_transformer import Transformer as PeriodPosTransformer

def get_model(model_type, **kwargs):
    """
    根据模型类型返回对应的模型实例。
    
    参数:
    - model_type: 模型类型，可选值为 "transformer", "conv_transformer", 
                  "conv_periodpos_transformer", "periodcpos_transformer"。
    - kwargs: 模型的初始化参数。
    """
    # 模型类型与类的映射
    model_classes = {
        "transformer": Transformer,
        "conv_transformer": ConvTransformer,
        "conv_periodpos_transformer": ConvPeriodPosTransformer,
        "periodcpos_transformer": PeriodPosTransformer,
    }

    # 检查模型类型是否支持
    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types are: {list(model_classes.keys())}"
        )

    # 返回对应的模型实例
    return model_classes[model_type](**kwargs)