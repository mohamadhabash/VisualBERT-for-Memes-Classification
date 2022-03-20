# VisualBERT-for-Memes-Classification
This repo contains VisualBERT implementation using Huggingface and PyTorch-Lightning for memes classification with the use of both text and images.

## How to use
1- clone the repo.
```bash
git clone https://github.com/mohamadhabash/VisualBERT-for-Memes-Classification.git
```
2- cd to the project and install the requirements.
```bash
pip3 install -r requirements.txt
```

3- put your images in `/Data/Images/` and your csv file in `/Data/`. You need to modify the code if do not use a `csv` file. 

4- Modify the code to include correct paths to your data. You can also tune hyper-parameters or change different stuff in the code.

5- run `visualbert.py` to start training the model.
```bash
python3 visualbert.py
```
6- You can also use the notebooks alone without other files. They can be found under `/notebooks`.
