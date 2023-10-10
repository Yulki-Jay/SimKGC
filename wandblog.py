import wandb
wandb.init(
    project="SimKGC-LinkPrediction",
    name="wn18rr",
    config={
        'model' : 'bert-base-uncased',
        "epochs": 50,
        "batch_size": 1024
    }
)

