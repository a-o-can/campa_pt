# functions for converting conditions to strings or one-hot encoded vectors
from campa.constants import get_data_config
import numpy as np
import logging

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def get_combined_one_hot(arrs):
    if len(arrs) != 2:
        raise NotImplementedError(f"combine {len(arrs)} arrs")
    mask = (~np.isnan(arrs[0][:,0]))&(~np.isnan(arrs[1][:,0]))
    n1 = arrs[0].shape[1]
    n2 = arrs[1].shape[1]
    targets = np.zeros(len(arrs[0]), dtype=np.uint8)
    targets[mask] = np.argmax(arrs[0][mask], axis=1) + n1*np.argmax(arrs[1][mask], axis=1)
    res = get_one_hot(targets, n1*n2)
    res[~mask] = np.nan
    return res

def convert_condition(arr, desc, one_hot=False, data_config=None):
    log = logging.getLogger('convert_condition')
    if data_config is None:
        log.warn("using default data config")
        data_config = get_data_config()
    # check if is a condition that can convert:
    if desc in data_config.CONDITIONS.keys():
        cur_conditions = data_config.CONDITIONS[desc]
        # need to go from str to numbers or the other way?
        if np.isin(arr, cur_conditions).any():
            log.info(f'Converting condition {desc} to numbers')
            conv_arr = np.zeros(arr.shape, dtype=np.uint8)
            for i,c in enumerate(cur_conditions):
                conv_arr[arr==c] = i
            if one_hot:
                conv_arr = get_one_hot(conv_arr, len(cur_conditions))
        else:
            log.info(f'Converting condition {desc} to strings')
            conv_arr = np.zeros(arr.shape, dtype=np.object)
            for i,c in enumerate(cur_conditions):
                conv_arr[arr==i]=c
        return conv_arr
    else:
        log.info(f'Not converting condition {desc} (is regression)')
        return arr

def process_condition_desc(desc):
    postprocess = None
    for proc in ['_one_hot', '_bin_3', '_lowhigh_bin_2', '_zscore']:
        if proc in desc:
            desc = desc.replace(proc, '')
            postprocess = proc[1:]
    return desc, postprocess  

def get_bin_3_condition(cond, desc, cond_params):
    """
    look for desc_bin_3_quantile kwarg specifying the quantile.
    If not present, calculate the quantiles based on cond.
    Then bin cond according to quantiles
    """
    # bin in .33 and .66 quantiles (3 classes)
    if cond_params.get(f'{desc}_bin_3_quantile', None) is not None:
        q = cond_params[f'{desc}_bin_3_quantile']
    else:
        q = np.quantile(cond, q=(0.33, 0.66))
        cond_params[f'{desc}_bin_3_quantile'] = list(q)
    cond_bin = np.zeros_like(cond).astype(int)
    cond_bin[cond > q[0]] = 1
    cond_bin[cond > q[1]] = 2
    return cond_bin, list(q)

def get_lowhigh_bin_2_condition(cond, desc, cond_params):
    # bin in 4 quantiles, take low and high TR cells (2 classes)
    # remainder of cells has nan values - can be filtered out later
    if cond_params.get(f'{desc}_lowhigh_bin_2_quantile', None) is not None:
        q = cond_params[f'{desc}_lowhigh_bin_2_quantile']
    else:
        q = np.quantile(cond, q=(0.25, 0.75))
        cond_params[f'{desc}_lowhigh_bin_2_quantile'] = list(q)
    cond_bin = np.zeros_like(cond).astype(int)
    cond_bin[cond > q[1]] = 1
    return cond_bin, list(q)

def get_zscore_condition(cond, desc, cond_params):
    # z-score TR
    if cond_params.get(f'{desc}_mean_std', None) is not None:
        tr_mean, tr_std = cond_params[f'{desc}_mean_std']
    else:
        tr_mean, tr_std = cond.mean(), cond.std()
        cond_params[f'{desc}_mean_std'] = [tr_mean, tr_std]
    cond = (cond - tr_mean) / tr_std
    return cond, [tr_mean, tr_std]