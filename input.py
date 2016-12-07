import reader


class Input(object):
    def __init__(self, config, data, name=None):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps
        self.input_data, self.targets \
            = reader.producer(data, config.batch_size, self.num_steps, name=name)