name = 'AsthmaResNet'

# Model:
model_cfg = dict(
    learning_rate = 1e-4,
    weight_decay = 1e-7
)

# Data:
data_module_cfg = dict(
    batch_size = 128,
    train_data = 'data/train_spiro.pkl',
    val_data = 'data/val_spiro.pkl',
    test_data = 'data/test_spiro.pkl',
    balance_train = False
)