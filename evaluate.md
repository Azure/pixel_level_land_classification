# Apply the trained model to new data

We will apply the trained model to an approximately 1 km x 1 km region centered at a point in Charlotte County, VA. This region is close to (but does not overlap with) the regions used to train the model; since the land type is very similar, we expect reasonable performance despite the short training time and limited training dataset.

## Submit the evaluation job

We will launch an evaluation job to apply our trained model to a specified region in our evaluation data. Before executing the command below, ensure that the `evaluation_job.json` file contained in this git repository has been downloaded to your computer and is available on the path, and that the training job has finished running.
```
az batchai job create -n evaluationjob -c evaluation_job.json -r batchaidemo --resource-group %AZURE_RESOURCE_GROUP% --location eastus
```
This job will take ~5 minutes to run; while waiting, you can read the section below for more information on what the job is doing. You can check on the job's progress using the following command:
```
az batchai job show -n evaluationjob --resource-group %AZURE_RESOURCE_GROUP%
```

When the job status changes to "Finished", the evaluation job is complete. You can also monitor the standard output and error messages as they're produced using the following commands:
```
az batchai job stream-file -d stdouterr -j evaluationjob -n stdout.txt -g %AZURE_RESOURCE_GROUP%
az batchai job stream-file -d stdouterr -j evaluationjob -n stderr.txt -g %AZURE_RESOURCE_GROUP%
```

To exit the streaming view, press Ctrl+C. You will be asked whether to terminate the job if it is still running: press "N" to indicate that you want the job to continue running.

## Understand the output of the evaluation job

### Description of the evaluation process

The [evaluation script](https://aiforearthcollateral.blob.core.windows.net/imagesegmentationtutorial/scripts/evaluate.py) extracts a 1 km x 1 km region from the evaluation data files. The center of the region is specified by lat-lon coordinates. (These coordinates, and other command-line arguments used by the script, are specified in the `evaluation_job.json` job config file.) The evaluation script loads the header information from the data files using GDAL, and interprets it to find the indices of the pixel that has the desired lat-lon coordinates. It then extracts the ground-truth labels and aerial imagery from the desired region. The extracted aerial image is "padded" by extending the boundaries of the extracted regions by 64 pixels in all directions.

Our trained model takes an input with dimensions 256 pixels x 256 pixels, and produces an output with dimensions 128 x 128 (corresponding to the center of the input region). To get output labels for the entire region of interest, the evaluation script must therefore use a tiling strategy, which is illustrated in the figure below. The model is first applied to the region at top-left of the extracted aerial image data. Since the aerial image data is appropriately padding, the model's output corresponds to the top-left corner of the actual region of interest. Next, we slide the model's input window 128 pixels to the right; the model's output for this tile corresponds to an adjacent location in the region of interest. Once we have covered all columns in a row, we shift the input window 128 pixels down and back to the left-hand boundary of the aerial image data. Finally, the model's predictions for all tiles are stitched together to produce a single image.

<img src="./outputs/tiling_strategy.PNG">

To visualize this output, we find the most likely label for each pixel and color-code the pixel accordingly. This visualization is simple to interpret but does not provide much information on the model's certainty in its predictions. If desired, you can modify the script to call `save_label_image()` using the argument `hard=False`: the color of each pixel will then be a blend of the label colors, weighted by the probability of each label.

### Accessing your evaluation job's output

We have applied our model to a 1 km x 1 km region centered on [a point in Charlotte County, VA](https://binged.it/2BcQfVQ). This region contains all four types of land cover that the model is able to predict: forested, herbaceous, barren/impervious, and water. The evaluation script will extract and save the [National Agricultural Imagery Program (NAIP) imagery](./outputs/NAIP.tif), [a pseudo-coloring Chesapeake Conservancy's ground-truth labels](./outputs/true_labels.tif), and [the trained model's predicted labels](./outputs/pred_labels.tif) for this region to the blob container attached to the Batch AI cluster. You can retrieve these outputs as follows:

1. Log into the [Azure Portal](https://portal.azure.com).
1. Use the search bar along the top of the screen to search for the storage account you created earlier. Click on the correct result to load the storage account's overview pane.
1. In the "Services" section, you will links labeled "Blobs" and "Files" inside of square bounding boxes. Click on "Blobs".
1. A list of blob containers will be displayed. Click on the container named "blobfuse" created by this tutorial.
1. A navigable directory structure will be displayed. You will find the output images from the evaluation job under the `evaluation_output` folder. 

    You can download files by clicking on the filename and then clicking "Download" in the pane that appears at right. You can also use this interface to upload new files, e.g. if you would like to modify the code in our training and evaluation scripts.

These extracted TIF images will each be ~12 MB in size and can be examined using a web browser (and most image/photo editing software).

### Review of sample results

Full-sized sample output images are provided in the [outputs folder](./outputs) of this repository; we provide a side-by-side, zoomed-out view below for easy comparison. While the tutorial creates a model trained for only one epoch, we also include an illustration of the model's output after 250 epochs to illustrate the accuracy that can ultimately be achieved:

<img src="./outputs/comparison_fullsize.PNG"/>

You'll notice that the model trained for 250 epochs has predicted the presence of a body of water (near top-center). This body of water is indeed present, demonstrating the potential of a sufficiently-trained model to suggest improvements on our ground-truth labels.

A copy of our sample trained model, `250epochs.model`, will be copied to your blob container's "models" folder during setup. If you like, you can modify `evaluation_job.json` to apply this model instead of the one you trained. Simply replace `trained.model` with `250epochs.model`, save the config file, and submit the job again with a unique job name.

## Next steps

When you are done, we recommend deleting all resources you created for this tutorial. Please see the instructions in the [setup section](./setup.md) of this tutorial. You may also be interested in our section on how to [scale this method](./scaling.md) for larger datasets and clusters.
