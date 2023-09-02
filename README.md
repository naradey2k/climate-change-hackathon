# Demo Web-Application for Cimate Change Hackathon
### Задача
> Алгоритм, который способен распознать мусор из 6 классов . В качестве входных данных вам будут предоставлены фотографии различного сырья.
---

### Модели

**Classification**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/naradey2k/climate-change-hackathon/blob/main/trainer.ipynb)

- модель [ResNet-50](https://pytorch.org/vision/stable/models.html) из зоопарка моделей torchvision + аугментации

### Ресурсы 
**Colab** с **NVIDIA Tesla V100**

Веса моделей (без поддержки ссылки):
[Google Drive](https://huggingface.co/spaces/dokster/Garbage_Chatter/blob/main/final_model_all.pt)

### Запуск демо
```python3
pip install torch, streamlit
git clone https://github.com/naradey2k/climate-change-hackathon.git
streamlit run main.py
```
