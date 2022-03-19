import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.functional import softmax
from torchmetrics.functional import f1, accuracy

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import VisualBertModel, VisualBertConfig

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score

from visual_embeds import *
from utils import *

RANDOM_SEED = 42
MAX_LEN = 64
N_CLASSES = 2
N_EPOCHS = 10
BATCH_SIZE = 32

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MemesDataset(Dataset):
    '''Wrap the tokenization process in a PyTorch Dataset, along with converting the labels to tensors'''
    
    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, visual_embeds):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.visual_embeds = visual_embeds
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row.misogynous

        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = torch.tensor(tokens["input_ids"]).flatten()
        attention_mask = torch.tensor(tokens["attention_mask"]).flatten()

        visual_embedding = self.visual_embeds[index].to('cpu')
        visual_attention_mask = torch.ones(visual_embedding.shape[:-1], dtype=torch.float)
        visual_token_type_ids = torch.ones(self.visual_embeds.shape[:-1], dtype=torch.long)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embedding=visual_embedding,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            labels=torch.tensor(labels).float()
        )


class MemesDataModule(pl.LightningDataModule):
    '''
    1- Split the dataset into training and validation dataset
    2- Create Dataloaders from Datasets (Divide data into batches)
    '''

    def __init__(self, df, tokenizer, visual_embeds, batch_size=32, max_len=64):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.visual_embeds = visual_embeds
  
    def setup(self, stage=None):
        self.dataset = MemesDataset(self.df, self.tokenizer, self.max_len, self.visual_embeds)
        self.train_dataset, self.val_dataset = train_test_split(self.dataset, test_size=0.1) # Split the dataset into training and validation datasets
  
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=3
        )
  
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers=3
        )
  
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers=3
        )


class MemesClassifier(pl.LightningModule):
  '''Wrap the training of VisualBERT model to classify memes'''

  def __init__(self, n_classes, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.configuration = VisualBertConfig.from_pretrained('uclanlp/visualbert-vqa-coco-pre', visual_embedding_dim=1024)
    self.model = VisualBertModel(self.configuration)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.CrossEntropyLoss()
    self.dropout = nn.Dropout(0.2)
    self.classifier = nn.Linear(self.model.config.hidden_size, n_classes)
  
  
  def forward(self, input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, labels=None):
    output = self.model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        visual_embeds=visual_embeds,
                        visual_attention_mask=visual_attention_mask,
                        visual_token_type_ids=visual_token_type_ids
    )
    
    output = self.dropout(output.pooler_output)
    output = self.classifier(output)

    loss = 0
    if labels is not None:
      loss = self.criterion(output, labels)

    return loss, output
  
  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    visual_embeds = batch['visual_embedding'].to(device)
    visual_attention_mask = batch['visual_attention_mask'].to(device)
    visual_token_type_ids = batch['visual_token_type_ids'].to(device)

    labels = batch['labels'].type(torch.LongTensor).to(device)
    
    loss, outputs = self(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, labels)
    self.log('train_loss', loss, prog_bar=True, logger=True)

    return {"loss":loss, 'predictions':outputs, 'labels':labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    visual_embeds = batch['visual_embedding']
    visual_attention_mask = batch['visual_attention_mask']
    visual_token_type_ids = batch['visual_token_type_ids'].to(device)
    labels = batch['labels'].type(torch.LongTensor).to(device)
    
    loss, outputs = self(input_ids, attention_mask, visual_embeds, visual_attention_mask, visual_token_type_ids, labels)
    self.log('val_loss', loss, prog_bar=True, logger=True)

    return loss
  
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=5e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.n_warmup_steps,
        num_training_steps=self.n_training_steps
    )

    return dict(
        optimizer=optimizer,
        lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
        )
    )


def train_model(model, df, tokenizer, visual_embeds):
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    
    logger = TensorBoardLogger("lightning_logs", name="memes-text")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

    data_module = MemesDataModule(
        df=df,
        tokenizer=tokenizer,
        visual_embeds=visual_embeds,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=10
    )

    trainer.fit(model, data_module)


def evaluate_model(test_dataset, checkpoint):
    trained_model = MemesClassifier.load_from_checkpoint(
        checkpoint,
        n_classes=2
    ).to(device)

    trained_model.eval()
    trained_model.freeze()

    predictions = []
    labels = []
    for item in tqdm(test_dataset):
        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device),
            item["visual_embedding"].unsqueeze(dim=0).to(device),
            item['visual_attention_mask'].unsqueeze(dim=0).to(device),
            item['visual_token_type_ids'].unsqueeze(dim=0).to(device)
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())
    
    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    _, preds = torch.max(torch.tensor(predictions), dim=1)
    
    f1_macro = f1_score(labels, preds , average="macro")
    f1_micro = f1_score(labels, preds , average="micro")
    accuracy = accuracy(predictions, labels)

    return f1_macro, f1_micro, accuracy
    

def main(): 
    df = pd.read_csv('./Data/your_training.csv')
    df.text = np.array([preprocess_text(text) for text in df.text])

    visual_embeds = generate_visual_embeds(
        images_file_names=df.file_name,
        images_path='./Data/Images', # You can put your images in this folder or change the path
        cfg_path='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    )
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    steps_per_epoch = 8000 // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 10

    model = MemesClassifier(
        n_classes=N_CLASSES,
        n_training_steps=total_training_steps,
        n_warmup_steps=warmup_steps,
    ).to(device)

    train_model(
        model=model, 
        df=df,
        tokenizer=tokenizer,
        visual_embeds=visual_embeds
    )

    test_df = pd.read_csv('./Data/your_test.csv')
    test_dataset = MemesDataset(
        test_df, 
        tokenizer, 
        MAX_LEN
    )

    f1_macro, f1_micro, accuracy = evaluate_model(
        test_dataset=test_dataset,
        checkpoint='./checkpoints/best-checkpoint.ckpt'
    )


if __name__ == '__main__':
    main()
