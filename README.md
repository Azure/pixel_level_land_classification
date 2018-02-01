# Pixel-level land use classification

This repository contains a tutorial illustrating how to create a deep neural network model that accepts an aerial image as input and returns a land cover label (forested, water, etc.) for every pixel in the image. Microsoft's [Cognitive Toolkit (CNTK)](https://www.microsoft.com/cognitive-toolkit/) is used to train and evaluate the model on an [Azure Geo AI Data Science Virtual Machine]() or an [Azure Batch AI](https://docs.microsoft.com/azure/batch-ai/) GPU cluster. The method shown here was developed in collaboration between the [Chesapeake Conservancy](http://chesapeakeconservancy.org/), [ESRI](https://www.esri.com), and [Microsoft Research](https://www.microsoft.com/research/) as part of Microsoft's [AI for Earth](https://www.microsoft.com/en-us/aiforearth) initiative.

We recommend budgeting two hours for a full walkthrough of this tutorial. The code, shell commands, trained models, and sample images provided here may prove helpful even if you prefer not to complete the walkthrough: we have provided explanations and direct links to these materials where possible.

## How to Get Started

The training and evaluation steps of this tutorial can be performed on either:
- an Azure Geo AI Data Science VM
    - Train a model on a data sample using Jupyter notebooks
    - Deploy the trained model directly in ESRI's ArcGIS Pro
- an Azure Batch AI GPU cluster
    - Set up your cluster and submit jobs to it from your command line
    - Learn how to scale to large clusters for faster training on larger datasets
    - (Optional) After training, download your model and use instructions from the Geo AI DSVM tutorials to deploy it in ArcGIS Pro

| Geo AI DSVM | Batch AI |
|-------------|----------|
|: [Create a Geo AI Data Science VM](./geoaidsvm/setup.md) :|: [Create a Batch AI cluster](./batchai/setup.md) :|
|: [Train a model in a Jupyter notebook](./geoaidsvm/Train_a_land_classification_model_from_scratch.ipynb) :|: [Train a model on the Batch AI cluster](./batchai/train.md) :|
|: [Evaluate the model using a Jupyter notebook](./geoaidsvm/Apply_trained_model_to_new_data.ipynb) :|: [Evaluate the model using a GPU cluster](./batchai/evaluate.md) :|
|: [Deploy your model in ArcGIS Pro](./geoaidsvm/Apply_trained_model_in_ArcGIS_Pro.ipynb) :|: [Learn how to scale up training](./batchai/scaling.md) :|

## Sample Output

This tutorial will train a pixel-level land use classifier for a single epoch: your model will produce results similar to bottom-left. By expanding the training dataset and increasing the number of training epochs, we achieved results like the example at bottom right. The trained model is accurate enough to detect some features, like the small pond at top-center, that were not correctly annotated in the ground-truth labels.

<img src="./outputs/comparison_fullsize.PNG"/>

## Related materials

- [Keynote demo from Microsoft Ignite](https://www.youtube.com/watch?time_continue=1&v=MUqo-lsAKgQ#t=23m46s)
- Blog post (forthcoming)
- Main [AI for Earth](https://www.microsoft.com/en-us/aiforearth) website
- [Publicity video on the Chesapeake Conservancy collaboration with Microsoft](http://chesapeakeconservancy.org/2017/07/10/microsoft-video-features-chesapeake-conservancy/)
- [Video clip showing real-time local application of the trained CNTK model through ESRI's ArcGIS software](https://www.youtube.com/watch?v=_iq-_K1OsMA)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
