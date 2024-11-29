# Large Language Model and State Space Model for Predicting Postoperative Recovery in Cerebral Hemorrhage

## Abstract

Intracerebral hemorrhage is an acute neurological disorder with high rates of disability and mortality. Accurate prediction of postoperative recovery is crucial for assisting physicians in personalized postoperative management. However, existing multimodal models have limitations in utilizing non-imaging clinical data, particularly in understanding its semantic features. To address this, we propose a novel multimodal prediction model that combines CT image features with personalized clinical data for precise recovery predictions three months after surgery. We first extract deep semantic features from clinical data using the ChatGPT language model and transform them into embeddings. Then, the MedMamba network, based on a state space model, extracts CT image features. Finally, an attention-based fusion module deeply integrates the CT and clinical data. Extensive experiments on real clinical datasets show a significant improvement in predicting the modified rankin scale score over existing models.

## Train
```python
python train.py
```
