def security_checks(config):
    if (config.training_settings['average_epochs'] <= 0) or (config.training_settings['average_epochs'] > config.training_settings['epochs']):
        raise RuntimeError(
            f"The number of epochs to compute an average model should be a value between 1 and the number of training epochs. You specified (average-epochs, training-epochs): {config.training_settings['average_epochs']}",
        )
