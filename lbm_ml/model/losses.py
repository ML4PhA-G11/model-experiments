import keras


def rmsre(y_true, y_pred, eps=1e-8):
    """Root Mean Square Relative Error — used as the training loss/metric.

    Divides by (y_true + eps) to make the error scale-invariant, so the network
    is penalised equally for errors in high- and low-density regions.
    """
    return keras.ops.sqrt(
        keras.ops.mean(
            keras.ops.square((y_true - y_pred) / (y_true + eps)), axis=-1
        )
    )
