def optimize_hyperparameters(hyperparam_space, train_and_evaluate):
    """
    Processes the hyperparameters in the given dictionary. For each parameter,
    it applies specific processing rules. For example, for 'learning_rate',
    it sets the value to the average if it's given as a tuple.

    Parameters:
    - hyperparam_space: A dictionary of hyperparameters.

    Returns:
    - A dictionary with processed hyperparameters.
    """
    processed_params = {}
    
    for key, value in hyperparam_space.items():
        if key == 'learning_rate' and isinstance(value, tuple):
            processed_params[key] = sum(value) / len(value)

    accuracy = train_and_evaluate( #alternatively, test_loss, accuracy = train_and_evaluate(
        batch_size=64,
        test_batch_size=1000,
        epochs=14,
        lr=processed_params['learning_rate'],
        gamma=0.7
    )
    
    print(f"Model trained and evaluated with accuracy: {accuracy}%")
    return processed_params
