import reader
import tensorflow as tf
from model import Model


class SmallConfig():
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    seq_length = 25


class MediumConfig():
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    seq_length = 25


class LargeConfig():
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    seq_length = 25


class TestConfig():
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    seq_length = 25

data_path = './'
model = "small"


def get_config():
  if model == "small":
    return SmallConfig()
  elif model == "medium":
    return MediumConfig()
  elif model == "large":
    return LargeConfig()
  elif model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", model)


def main():
    raw_data = reader.raw_data(data_path)
    train_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    num_batches = int(train_data.size / (config.batch_size * config.seq_length))
    x_batches, y_batches = reader.create_batches(train_data, num_batches=num_batches,
                          batch_size=config.batch_size, seq_length=config.seq_length)


    model = Model()





if __name__ == '__main__':
    main()