def get_efficientnet_v2_hyperparam(model_name, stage=4):
    # batch_size, train_size, dropout, randaug, mixup (end)
    if 'efficientnet_v2_s' in model_name:
        init = 384, 128, 0.1, 5, 0
        end, eval_size = (128, 300, 0.2, 10, 0), 384
    elif 'efficientnet_v2_m' in model_name:
        init = 128, 0.1, 5, 0
        end, eval_size = (384, 0.3, 15, 0.2), 480
    elif 'efficientnet_v2_l' in model_name:
        init = 128, 0.1, 5, 0
        end, eval_size = (384, 0.4, 20, 0.5), 480
    elif 'efficientnet_v2_xl' in model_name:
        init = 128, 0.1, 5, 0
        end, eval_size = (384, 0.4, 20, 0.5), 512
    return get_param_per_stage(init, end, stage), eval_size,


def get_param_per_stage(init, end, n):
    stages = []
    for i in range(n):
        stages.append([s + (e - s) / (n - 1) * i for s, e in zip(init, end)])
    return stages