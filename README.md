<img src="./outputs/comparison_fullsize.PNG"/>

## Get Started

1. [Install prerequisities and set up Azure resources](./setup.md)

    After you complete this section, you'll have a Batch AI GPU cluster loaded with the sample data and scripts required by this tutorial.
1. [Train a land classification model from scratch](./train.md)

    In this section, you'll produce a trained CNTK model that you can use anywhere for pixel-level land cover prediction.
1. [Apply your trained model to new aerial images](./evaluate.md)

    You'll predict land use on a 1 km x 1 km region not previously seen during training, and examine your results in full color.
1. [Learn more about scaling up training](./scaling.md)

    You'll see how we scaled this solution to train a model for 250 epochs, with 139x as much data and 74x as many GPUs, in a little over two hours!

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
