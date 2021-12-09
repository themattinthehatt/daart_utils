import pandas as pd
import time

from daart.eval import get_precision_recall


def random_forest_search(
        features_train, targets_train, features_val=None, targets_val=None,
        n_estimators_list=[500], min_samples_leaf_list=[1], criterion_list=['entropy'],
        rng_seed=0, verbose=True
):
    """Fit a series of random forest binary classifiers.

    Parameters
    ----------
    features_train : np.ndarray
        shape (n_samples, n_features)
    targets_train : np.ndarray
        shape (n_samples,)
    features_val : np.ndarray, optional
        shape (n_samples, n_features); if None, results dict will contain evaluation on train data
    targets_val : np.ndarray, optional
        shape (n_samples,); if None, results dict will contain evaluation on train data
    n_estimators_list : list of int, optional
        number of trees in forest
    min_samples_leaf_list : list of int, optional
        minimum samples per leaf node
    criterion_list : list of str, optional
        splitting criterion; 'entropy' | 'gini'
    rng_seed : int, optional
        random seed for random forest
    verbose : bool, optional
        print info along the way

    Returns
    -------
    tuple
        list: models
        pd.DataFrame: eval results

    """
    from sklearn.ensemble import RandomForestClassifier
    metrics = []
    models = []
    index = 0
    for n_estimators in n_estimators_list:
        for min_samples_leaf in min_samples_leaf_list:
            for criterion in criterion_list:

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features='sqrt',
                    criterion=criterion,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=True,
                    n_jobs=-1,
                    random_state=rng_seed,
                )
                if verbose:
                    print(model)
                t_beg = time.time()
                model.fit(features_train, targets_train)
                models.append(model)
                t_end = time.time()
                if verbose:
                    print('fitting time: %1.2f sec' % (t_end - t_beg))

                if features_val is not None and targets_val is not None:
                    preds = model.predict(features_val)
                    results = get_precision_recall(targets_val, preds, background=None)
                else:
                    print('no validation data; evaluating on train data')
                    preds = model.predict(features_train)
                    results = get_precision_recall(targets_train, preds, background=None)

                metrics.append(pd.DataFrame({
                    'n_estimators': n_estimators,
                    'min_samples_leaf': min_samples_leaf,
                    'criterion': criterion,
                    'precision': results['precision'][-1],
                    'recall': results['recall'][-1],
                    'f1': results['f1'][-1],
                }, index=[index]))
                index += 1

    metrics = pd.concat(metrics)
    return models, metrics


def logistic_regression_search(
        features_train, targets_train, features_val=None, targets_val=None, l2_reg_list=[0],
        max_iter=5000, verbose=True
):
    """Fit a series of logistic regression binary classifiers.

    Parameters
    ----------
    features_train : np.ndarray
        shape (n_samples, n_features)
    targets_train : np.ndarray
        shape (n_samples,)
    features_val : np.ndarray, optional
        shape (n_samples, n_features); if None, results dict will contain evaluation on train data
    targets_val : np.ndarray, optional
        shape (n_samples,); if None, results dict will contain evaluation on train data
    l2_reg_list : list of int, optional
        l2 regularization parameter on weights
    max_iter : int, optional
        max number of iterations
    verbose : bool, optional
        print info along the way

    Returns
    -------
    tuple
        list: models
        pd.DataFrame: eval results

    """
    from sklearn.linear_model import LogisticRegression
    metrics = []
    models = []
    index = 0
    for l2_reg in l2_reg_list:

        model = LogisticRegression(penalty='l2', max_iter=max_iter, C=l2_reg)
        if verbose:
            print(model)
        t_beg = time.time()
        model.fit(features_train, targets_train)
        models.append(model)
        t_end = time.time()
        if verbose:
            print('fitting time: %1.2f sec' % (t_end - t_beg))

        if features_val is not None and targets_val is not None:
            preds = model.predict(features_val)
            results = get_precision_recall(targets_val, preds, background=None)
        else:
            print('no validation data; evaluating on train data')
            preds = model.predict(features_train)
            results = get_precision_recall(targets_train, preds, background=None)

        metrics.append(pd.DataFrame({
            'l2_reg': l2_reg,
            'precision': results['precision'][-1],
            'recall': results['recall'][-1],
            'f1': results['f1'][-1],
        }, index=[index]))
        index += 1

    metrics = pd.concat(metrics)
    return models, metrics


def mlp_search(
        features_train, targets_train, features_val=None, targets_val=None,
        hidden_layers_list=[32], hidden_units_list=[32, 64], activation_list=['tanh'],
        l2_reg_list=[1e-5], learning_rate_list=[1e-3], max_iter=5000, batch_size=256, rng_seed=0,
        verbose=True
):
    """Fit a series of logistic regression binary classifiers.

    Parameters
    ----------
    features_train : np.ndarray
        shape (n_samples, n_features)
    targets_train : np.ndarray
        shape (n_samples,)
    features_val : np.ndarray, optional
        shape (n_samples, n_features); if None, results dict will contain evaluation on train data
    targets_val : np.ndarray, optional
        shape (n_samples,); if None, results dict will contain evaluation on train data
    hidden_layers_list : list of int, optional
        number of hidden layers
    hidden_units_list : list of int, optional
        hidden units per layer; constrained to be the same number for each layer
    activation_list : list of str, optional
        'relu' | 'tanh' | 'logistic' | 'identity' | 'softmax'
    l2_reg_list : list of int, optional
        l2 regularization parameter on weights
    learning_rate_list : list of float, optional
        initial adam learning rate
    max_iter : int, optional
        max number of iterations
    batch_size : int, optional
        number of samples in sgd batch
    rng_seed : int, optional
        random seed for mlp model fitting
    verbose : bool, optional
        print info along the way

    Returns
    -------
    tuple
        list: models
        pd.DataFrame: eval results

    """
    from sklearn.neural_network import MLPClassifier
    metrics = []
    models = []
    index = 0
    for hidden_layers in hidden_layers_list:
        for hidden_units in hidden_units_list:
            for activation in activation_list:
                for l2_reg in l2_reg_list:
                    for learning_rate in learning_rate_list:
                        model = MLPClassifier(
                            hidden_layer_sizes=[hidden_units] * hidden_layers,
                            activation=activation,
                            alpha=l2_reg,
                            learning_rate_init=learning_rate,
                            batch_size=batch_size,
                            max_iter=max_iter,
                            random_state=rng_seed,
                        )

                        if verbose:
                            print(model)
                        t_beg = time.time()
                        model.fit(features_train, targets_train)
                        models.append(model)
                        t_end = time.time()
                        if verbose:
                            print('fitting time: %1.2f sec' % (t_end - t_beg))

                        if features_val is not None and targets_val is not None:
                            preds = model.predict(features_val)
                            results = get_precision_recall(targets_val, preds, background=None)
                        else:
                            print('no validation data; evaluating on train data')
                            preds = model.predict(features_train)
                            results = get_precision_recall(targets_train, preds, background=None)

                        metrics.append(pd.DataFrame({
                            'hidden_layers': hidden_layers,
                            'hidden_units': hidden_units,
                            'activation': activation,
                            'alpha': l2_reg,
                            'learning_rate': learning_rate,
                            'precision': results['precision'][-1],
                            'recall': results['recall'][-1],
                            'f1': results['f1'][-1],
                        }, index=[index]))
                        index += 1

    metrics = pd.concat(metrics)
    return models, metrics
