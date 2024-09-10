name = 'AsthmaPhE-T'

# Model:
model_cfg = dict(
    ckpt_phet = 'ckpts/PhE-T/v0/best-epoch=3-step=3842.ckpt',
    ckpt_resnet = 'ckpts/AsthmaResNet/v0/best-epoch=5-step=9795.ckpt',
    n_embeds = 1,
    freeze_phet = True,
    freeze_resnet = True,
    learning_rate = 1e-4,
    warmup_steps = 0,
    weight_decay = 0.01
)

# Data:
num_features = [
    'BMI',
    'HDL cholesterol',
    'LDL cholesterol',
    'Total cholesterol',
    'Triglycerides',
    'Diastolic blood pressure'
]
cat_features = [
    'Age',
    'Sex',
    'Ever smoked',
    'Snoring',
    'Insomnia',
    'Daytime napping',
    'Chronotype',
    'Sleep duration',
]
data_module_cfg = dict(
    batch_size = 256,
    tab_train = 'data/train.csv',
    tab_val = 'data/val.csv',
    tab_test = 'data/test.csv',
    spiro_train = 'data/train_spiro.pkl',
    spiro_val = 'data/val_spiro.pkl',
    spiro_test = 'data/test_spiro.pkl',
    ckpt_tokenizer = model_cfg['ckpt_phet'],
    num_features = num_features,
    cat_features = cat_features,
)