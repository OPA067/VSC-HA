<div align="left">
  
# 【Paper Title】VSC-HA: Text-Video Retrieval with Video Semantic Compression and Hierarchical Alignment

In this paper, we propose a method for semantic compression of video and hierarchical granularity alignment, making the video more flexible and effective in matching with text.



## 📣 Updates
* **[2024/08/20]**: We have released the complete code of the model, including both training and testing code.
* **[2024/09/01]**: We have provided a comprehensive description of the dataset, model deployment, training methods, and testing methods.

## ⚡ Framework
<div align="center">
<img src="/images/Fig-Framework.png" width="800px">  
</div>
<div align="center">
<img src="/images/Fig-Alignment.png" width="800px">  
</div>

## 😍 Visualization

### Example 1
<div align=center>
<img src="/images/result-3.png" width="800px">
</div>

### Example 2
<div align=center>
<img src="/images/result-1.png" width="800px">
</div>

## 🚀 Quick Start
### Setup

#### Setup code environment
```shell
conda create -n VSC-HA python=3.8
conda activate VSC-HA
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Download CLIP Model

Download ViT-B/32 Model: [ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32])

Download ViT-B/16 Model:  [ViT-B/16](https://huggingface.co/openai/clip-vit-base-patch16])

#### Download Datasets

<div align=center>

|       Datasets        |                             Download Link                              |
|:---------------------:|:----------------------------------------------------------------------:|
|        MSRVTT         |      [Download](http://ms-multimedia-challenge.com/2017/dataset)       |  
|         MSVD          | [Download](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) | 
| ActivityNet |           [Download](http://activity-net.org/download.html)            | 
| Charades |         [Download](https://github.com/activitynet/ActivityNet)         |  
| DiDeMo |       [Download](https://github.com/LisaAnne/LocalizingMoments)        | 
| VATEX |                              [Download](https://eric-xw.github.io/vatex-website/download.html)                              | 

</div>

#### Training
Run the following training code to resume the above results. Take MSRVTT as an example.

```python
python train.py  --exp_name=MSRVTT-train --save_memory_mode --dataset_name=MSRVTT --num_epochs=5 --log_step=1 --evals_per_epoch=1 --batch_size=32 --num_workers=8 --videos_dir=MSRVTT/videos/ --noclip_lr=3e-5  
```

#### Testing
```python
python test.py  --exp_name=MSRVTT-test --save_memory_mode --dataset_name=MSRVTT --batch_size=32 --num_workers=8 --videos_dir=MSRVTT/videos/ --noclip_lr=3e-5 --load_epoch=0 --datetime=test
```

## 🎗️ Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip/), [X-Pool](https://github.com/layer6ai-labs/xpool), [T-Mass](https://github.com/Jiamian-Wang/T-MASS-text-video-retrieval). We sincerely appreciate for their contributions.
