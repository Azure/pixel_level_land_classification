# Pixel-level land use classification

This repository contains a tutorial illustrating how to create a deep neural network model that accepts an aerial image as input and returns a land cover label (forested, water, etc.) for every pixel in the image. Microsoft's [Cognitive Toolkit (CNTK)](https://www.microsoft.com/cognitive-toolkit/) is used to train and evaluate the model on an [Azure Batch AI](https://docs.microsoft.com/azure/batch-ai/) GPU cluster. The method shown here was developed in collaboration between the [Chesapeake Conservancy](http://chesapeakeconservancy.org/), [ESRI](https://www.esri.com), and [Microsoft Research](https://www.microsoft.com/research/) as part of Microsoft's [AI for Earth](https://www.microsoft.com/en-us/aiforearth) initiative.

We recommend budgeting two hours for a full walkthrough of this tutorial. The code, shell commands, trained models, and sample images provided here may prove helpful even if you prefer not to complete the tutorial: we have provided explanations and direct links to these materials where possible.

## How to Get Started

1. [Install prerequisites and set up Azure resources](./setup.md)

    After you complete this section, you'll have created a Batch AI GPU cluster loaded with the sample data and scripts required by this tutorial.
1. [Train a land classification model from scratch](./train.md)

    In this section, you'll produce a trained CNTK model that you can use anywhere for pixel-level land cover prediction.
1. [Apply your trained model to new aerial images](./evaluate.md)

    You'll predict land use on a 1 km x 1 km region not previously seen during training, and examine your results in full color.
1. [Learn more about scaling up training](./scaling.md)

    You'll see how we scaled this solution to train a model for 250 epochs, on a much larger dataset, in a little over two hours!
    
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
