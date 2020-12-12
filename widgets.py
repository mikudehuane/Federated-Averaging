import os
import traceback


def get_spare_dir(f_dir, c_dir, new=True):
    """
    :param new: seek for new path
    :param f_dir: father directory
    :param c_dir: child directory prefix
    :return: a created idle directory like f_dir/c_dir0
    """
    test_num = 0
    while os.path.exists(os.path.join(f_dir, c_dir + str(test_num))):
        test_num += 1
    if not new:
        test_num -= 1
    assert test_num >= 0
    idle_path = os.path.join(f_dir, c_dir + str(test_num))
    os.makedirs(idle_path, exist_ok=True)
    return idle_path


def init_pars(dst, src):
    """
    init self.pars in __init__
    :param dst: self.pars
    :param src: pars delivered
    :return: None
    """
    try:
        for k in src:
            if k not in dst:
                print(f"Invalid parameter {k}")
                raise AssertionError
            dst[k] = src[k]
    except AssertionError as e:
        traceback.print_exc()
        raise e


def zero_model(model):
    for par in model.parameters():
        par.data.sub_(par.data)
    return model


def aggregate_model(global_model, local_model, global_weight, local_weight):
    """
    :param local_weight:
    :param global_weight:
    :param global_model:
    :param local_model:
    :return:
    """
    for ind, par in enumerate(global_model.parameters()):
        par.data *= global_weight
        par.data.add_(local_model[ind].data * local_weight)


def clamp(number, low, high):
    if number < low:
        number = low
    if number > high:
        number = high
    return number
