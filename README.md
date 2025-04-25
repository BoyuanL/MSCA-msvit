# Enhanced Kinship Verification via Context-Aware Multi-Scale Transformer

**This is the official implementation of the paper**: Enhanced Kinship Verification via Context-Aware Multi-Scale Transformer. Submitted to ***The Visual Computer***. 

Note: The cited paper is currently under peer-review at The Visual Computer. This citation is provisional and subject to the journal's acceptance decision.

## Installation

Our code has been tested in Python 3.7 and PyTorch 1.13.1+cu117 environments. Please set up your environment according to the official instructions. See other required packages in `requirements.txt`.



## Data Preparation

We trained and evaluated our method on publicly available Families In the Wild(FIW), KinFaceW-I, and KinFaceW-II datasets. Download and extract the datasets from their official sources, then organize the data with the following structure:

- [Families In the Wild](https://github.com/visionjo/fiw)

      |--data                         
          |--FIW           
              |--pairs
              |--Test
              |--Train
              |--Validation    
          |--KinFaceW-I
          |--KinFaceW-II
  
- [KinFaceW-I, and KinFaceW-II](https://www.kinfacew.com/datasets.html)

  

## Pre-trained Weights

Pre-trained weights are available for download via this [Baidu Drive link](https://pan.baidu.com/s/1wUsqQJgslMlOEzL83XBsaA). Extraction code: 3vn8.



## Model Training

```python
python main.py --mode test --data_dir data/FIW --backbone vit --checkpoint checkpoints/model_best.pth --output_csv results/predictions.csv
```



## Model Testing

```python
python main.py --mode test --data_dir data/FIW --backbone vit --checkpoint checkpoints/model_best.pth --output_csv results/predictions.csv
```

### Citation

If you use this work, please cite our paper:

```bibtex
@article{zhu2025enhanced,  
  title={Enhanced Kinship Verification via Context-Aware Multi-Scale Transformer},  
  author={Zhu, Xiaoke and Li, Boyuan and Chen, Xiaopan and Qi, Fumin and Yuan, Caihong and Jing, Xiao-Yuan},    
  year={2025}, 
  note={Submitted to The Visual Computer},
}  
```