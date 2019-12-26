import argparse

parser = argparse.ArgumentParser()

# parser.add_argument("-a", "--action", default='train', help="test/train")
parser.add_argument("-tv", "--task_version", default='rt', help="rt/fd")
parser.add_argument("-e_num", "--epoch_num", default=10e3, help="num of epochs for training")

args = parser.parse_args()

SGD_p = {
    'lr': 0.01,
    'max_grad_norm': 1,
    'vanish_grad_reg': 2,
    'tau': 100,
    'train_t_step': 20,
    'test_t_step': 0.5,
    'ini_spe_r': 1.5,
    'minibatch_size': 20,
    'baseline_input': 0.2,
    'input_noise_std': 0.01,
    'rr_noise_std': 0.15,
    'mini_w_threshold': 10**-4
}


