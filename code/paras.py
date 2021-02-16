from param_handler import yaml_loader


def load_params(para_file):
    if para_file is None:
        para = set_default_params()
    else:
        para = yaml_loader().load(para_file)
    para['rand_seed'] = 1
    para['category_list'] = ['Food', 'Shop & Service', 'Travel & Transport',
                             'College & University', 'Nightlife Spot',
                             'Residence', 'Outdoors & Recreation',
                             'Arts & Entertainment',
                             'Professional & Other Places']
    return para


def set_default_params():
    pd = dict()
    pd['data_dir'] = '../data/toy/'
    pd['tweet_file'] = pd['data_dir'] + 'input/tweets.txt'
    pd['poi_file'] = pd['data_dir'] + 'input/pois.txt'
    pd['result_dir'] = pd['data_dir'] + 'output/'
    pd['model_dir'] = pd['data_dir'] + 'model/'
    pd['model_embeddings_dir'] = pd['result_dir'] + 'embeddings/'
    pd['model_pickled_path'] = pd['model_dir'] + 'pickled.model'

    pd['load_existing_model'] = False
    pd['voca_min'] = 0
    pd['voca_max'] = 20000
    pd['dim'] = 10
    pd['negative'] = 1
    pd['alpha'] = 0.02  # learning rate
    pd['epoch'] = 1
    pd['nt_list'] = ['w', 'l', 'c']
    pd['predict_type'] = ['w', 'l', 'c', 'p']
    # used for efficiency reason (requested by fast k-nearest-neighbor search)
    pd['kernel_nb_num'] = 1
    pd['kernel_bandwidth_l'] = 0.001
    pd['kernel_bandwidth_t'] = 1000.0
    pd['test_batch_num'] = 2
    pd['grid_len'] = 0.002
    pd['update_strategy'] = 'iteration-based'
    pd['regu_weight'] = 0.1
    pd['decay_rate'] = 0.001
    pd['update_tweets'] = 0
    pd['embed_algo'] = 'react'
    return pd
