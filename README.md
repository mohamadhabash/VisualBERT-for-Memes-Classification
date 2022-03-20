# VisualBERT-for-Memes-Classification
# Overview
This repo contains VisualBERT implementation using Huggingface and PyTorch-Lightning for memes classification with the use of both text and images. VisualBERT consists of a stack of transformer layers similar to BERT architecture to prepare embeddings for image-text pairs. BERT tokenizer is used as a text encoder. For images, a custom pre-trained object detector must be used to extract regions and bounding boxes, which will be fed to the model as visual embeddings. `Detectron2` is used in this repo to generate the visual embeddings using `MaskRCNN+ResNet-101+FPN` model checkpoint.

## How to use
1- clone the repo.
```bash
git clone https://github.com/mohamadhabash/VisualBERT-for-Memes-Classification.git
```
2- `cd` to the project and install the requirements.
```bash
pip3 install -r requirements.txt
```

3- put your images in `/Data/Images/` and your csv file in `/Data/`. You need to modify the code if you do not use a `csv` file. 

4- Modify the code to include correct paths to your data. You can also tune hyper-parameters or change different stuff in the code.

5- run `visualbert.py` to start training the model.
```bash
python3 visualbert.py
```
6- You can also use the notebooks alone without other files. They can be found under `/notebooks`.

## References
1- <a href="https://github.com/uclanlp/visualbert">VisualBERT Repository</a>

2- <a href="https://huggingface.co/docs/transformers/model_doc/visual_bert">Huggingface VisualBERT Documentation</a>

3- <a href="https://github.com/PyTorchLightning/pytorch-lightning">PyTorch Lightning</a>

4- <a href="https://github.com/pytorch/pytorch">PyTorch</a>

5- <a href="https://github.com/facebookresearch/detectron2">Detectron2</a>
