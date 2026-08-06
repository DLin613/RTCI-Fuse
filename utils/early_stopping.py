import copy


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='loss', eval_freq=1):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.eval_freq = eval_freq

        self.best_loss = float('inf')

        self.counter = 0
        self.best_model_state = None
        self.epoch_count = 0

    def check_loss(self, current_loss, model):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def __call__(self, model, tqdm_obj=None, val_loader=None, current_loss=None):
        self.epoch_count += 1

        if self.mode == 'loss':
            if current_loss is None:
                raise ValueError("loss模式需要提供current_loss参数")

            should_stop = self.check_loss(current_loss, model)

            if tqdm_obj is not None:
                tqdm_obj.write(f"Epoch[{self.epoch_count}] best loss: {self.best_loss:.6f}, current: {current_loss:.6f}")

            return should_stop