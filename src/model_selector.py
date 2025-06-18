def select_best_model(results):
    # results is a list of tuples: (model_name, model_obj, score_dict)
    sorted_models = sorted(results, key=lambda x: x[2]['mae'])
    return sorted_models[0]
