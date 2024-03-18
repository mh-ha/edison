import torch
from torch import utils
import lightning as L
from networks import LD4LG


model = LD4LG()
dataset = None #TODO: Lightning supports ANY iterable (DataLoader, numpy, etc…) for the train/val/test/predict splits.
train_loader = utils.data.DataLoader(dataset, batch_size=32)

trainer = L.Trainer(max_epochs=10, callbacks=[L.ModelCheckpoint(dirpath="lightning_logs")])
trainer.fit(model=model, train_dataloaders=train_loader)



# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LD4LG.load_from_checkpoint(checkpoint)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

