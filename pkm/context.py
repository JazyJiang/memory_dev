class PKMContext:
    _current_group_ids = None
    _current_epoch = 0
    _warmup_epochs = 0
    _is_training = True

    @classmethod
    def set_group_ids(cls, ids):
        cls._current_group_ids = ids

    @classmethod
    def get_group_ids(cls):
        return cls._current_group_ids

    @classmethod
    def set_epoch(cls, epoch):
        cls._current_epoch = epoch

    @classmethod
    def get_epoch(cls):
        return cls._current_epoch

    @classmethod
    def set_warmup_epochs(cls, epochs):
        cls._warmup_epochs = epochs

    @classmethod
    def is_warmup(cls):
        return cls._is_training and cls._current_epoch < cls._warmup_epochs

    @classmethod
    def set_training(cls, mode):
        cls._is_training = bool(mode)

    @classmethod
    def is_training(cls):
        return bool(cls._is_training)