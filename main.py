import argparse
import rnn.rnn_model as rnn

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--action", default='train', help="test/train")
parser.add_argument("-tv", "--task_version", default='rt', help="rt/fd")
parser.add_argument("--init_state_trainable", default=True, help="Set if init sate trainable")
parser.add_argument("-e_num", "--epoch_num", default=50, help="max num of epochs for training")
parser.add_argument("-e_size", "--epoch_size", default=1000, help="num of trails in one epoch for training")
parser.add_argument("--model_date", default=None)
# parser.add_argument("--model_date", default='20200111-133345')

args = parser.parse_args()
args = vars(args)

SGD_p = {
    'lr': 0.1,  # TODO: origin is 0.01
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
    'mini_w_threshold': 10 ** -7  # TODO: origin is 10**-4
}

if __name__ == '__main__':
    ei_rnn = rnn.SimpleEIRNN(args)
    ei_rnn.build()
    if args['action'] == 'train':
        ei_rnn.train()
    else:
        ei_rnn.test()
