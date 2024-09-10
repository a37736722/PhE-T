name = 'PhE-T'

# Model:
model_cfg = dict(
    phet_config = {
        'n_layers': 12,
        'n_heads': 12,
        'h_dim': 768,
        'ln_eps': 1e-12,
        'dropout': 0.1
    },
    learning_rate = 1e-4,
    adamw_epsilon = 1e-6,
    adamw_betas = (0.9, 0.98),
    warmup_steps = 1000,
    weight_decay = 0.01,
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
diseases = [
    'Asthma',
    'Cataract',
    'Diabetes',
    'GERD',
    'Hay-fever & Eczema',
    'Major depression',
    'Myocardial infarction',
    'Osteoarthritis',
    'Pneumonia',
    'Stroke'
]
data_module_cfg = dict(
    train_data = 'data/train.csv',
    val_data = 'data/val.csv',
    test_data = 'data/test.csv',
    num_features =  num_features,
    cat_features = cat_features + diseases,
    n_bins = 100,
    binning = 'quantile',
    batch_size = 256,
    mhm_probability = 0.15,
)

