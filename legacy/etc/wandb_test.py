import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="experiment_edison",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.002,
        "architecture": "test",
        "dataset": "test",
        "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"test_1": acc, "test_2": loss})

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()
