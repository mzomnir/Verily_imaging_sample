# Lung Cancer Mutation Detection

This repo contains programs for training, evaluating, and applying a deep neural network for lung cancer mutation detection. Our goal is to infer genotype from phenotype: given a digitized slide scan image of cancerous lung tissue, infer whether the cancer contains mutations in a given gene (we start with KRAS and EGFR). All data is obtained from Massachusetts General Hospital's Center for Integrated Diagnostics and is not publicly available.  A small sample of data is included in this repo. 

During preprocessing, we slice each image into multiple tiles ("chunks"), permute it (e.g. rotate, flip), and augment it (e.g. with active contouring).  We continue to experiment with new preprocessing techniques to improve training performance.

We have so far experimented with three models for training: a basic CNN and two different versions of Inception ResNet.  We continue to experiment with tweaking the models (e.g. with regularization, dropout, inverse class weighting, etc.).  During training, the model processes each image chunk individually, assinging a binary prediction of whether there is/not a mutation in the given gene for the tumor tissues in the image chunk.  We train separate models for each gene (KRAS and EGFR).  Importantly, all of our raw training images contain cancer, and our models assume the presence of cancer in the images. Many other researchers have worked on the problem of inferring whether a slide image contains cancerous tissue.  We take as a given that our slide images contain cancer and then try to infer whether the tumor DNA contains mutation(s) in a given gene.  

We realized early on that, while the genomic labels for our raw (un-chunked) training images were unambiguous, it is very possible that after subdividing those images into chunks, some of those chunks will not contain any cancerous tissue. We have experimented with several approaches to dealing with this, including elmininating chunks containing more than a certain threshold of whitespace, filtering our training dataset for images in which cancerous tissue occupies more than a certain threshold volume of the slide, and having pathologists manually outline cancerous regions in the training images.

We have thus far evaluated training perfomance at the image-chunk level (rather than the whole-image or patient level), using the patient-level genomic label for every associated image chunk.  This approach will be affected by the aforementioned problem of some image sub-chunks containing little or no cancerous tissue.  Since our ultimate goal is to infer genotypes at the patient-level, rather than the whole-image or image-chunk level, our final model will contain a voting step that assigns a label to each whole-image based on a consensus of predictions on its image-chunks, followed by a final voting step that assigns a final patient-level label based on a consensus of the whole-images for the given patient.

## Prerequisites

- Set up `data` directory in the root of the project with all training images in `data/original_images` and a csv version of Training_ML_clean.xlsb as `data/Training_ML_clean.csv`.
- Create a virtualenv and install all requirements in `requirements.txt`.
- Install CUDA 9.0 and cuDNN 7.0.

## Preprocess Images

Run:
```python
python mutation_detection/preprocess_data.py --config_name inception_resnet_1
```

The config name `inception_resnet_1` is one existing configuration, but we can create new configurations in the `mutation_detection/models` directory. We just copy one configuration to a new file and make whatever edits we'd like to have in the new configuration.

## Training

Run:
```python
python mutation_detection/train.py --config_name inception_resnet_1
```

Again, the config name is something we can freely change.

## Coming Soon

- [ ] Model checkpointing (checkpoint callback is set up for parallel training)
- [ ] Experiment logging in sqlite
- [ ] Experiment result viz notebook (include dated notes and viz for experiments)
- [ ] Separate experiment configurations for the different tasks (only therapeutic detection available for now)
- [ ] Support for additional inputs other than just image data (e.g. acquisition method)
- [ ] Support for second test dataset provided by MGH.
