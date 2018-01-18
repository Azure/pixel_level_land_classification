# Apply the trained model to new data

We will apply the trained model to an approximately 1 km x 1 km region centered at a point in Charlotte County, VA. This region is close to (but does not overlap with) the regions used to train the model; since the land type is very similar, we expect reasonable performance despite the short training time and limited training dataset.

## Submit the evaluation job

We will launch an evaluation job to apply our trained model to a specified region in our evaluation data. Before executing the command below, ensure that the `evaluation_job.json` file contained in this git repository has been downloaded to your computer and is available on the path, and that the training job has finished running.
```
az batchai job create -n evaluationjob -c evaluation_job.json -r batchaidemo
```
This job will take ~5 minutes to run; while waiting, you can read the section below for more information on what the job is doing. You can check on the job's progress using the following command:
```
az batchai job show -n evaluationjob
```

When the job status changes to "Finished", the evaluation job is complete.

## Understand the output of the evaluation job

### Description of the evaluation process

The [evaluation script](https://aiforearthcollateral.blob.core.windows.net/imagesegmentationtutorial/scripts/evaluate.py) extracts a 1 km x 1 km region of interest whose center is specified by lat-lon coordinates. (These coordinates, and other command-line arguments used by the script, are specified in the `evaluation_job.json` job config file.) 

If you like, you can modify `evaluation_job.json` to use a sample model that we prepared by training on a larger dataset for 250 epochs. Simply replace `trained.model` with `250_epochs.model`, save the config file, and submit the job again with a unique job name. (Your previous output images will be overwritten unless you also change the name of your output folder.)

### Accessing your evaluation job's output

We have applied our model to a 1 km x 1 km region centered on [a point in Charlotte County, VA](https://binged.it/2BcQfVQ). This region contains all four types of land cover that the model is able to predict: forested, herbaceous, barren/impervious, and water. The evaluation script will extract and save the [National Agricultural Imagery Program (NAIP) imagery](./outputs/NAIP.tif), [a pseudo-coloring Chesapeake Conservancy's ground-truth labels](./outputs/true_labels.tif), and [the trained model's predicted labels](./outputs/pred_labels.tif) for this region to the blob container attached to the Batch AI cluster. You can retrieve these outputs as follows:

1. Log into the [Azure Portal](https://portal.azure.com).
1. Use the search bar along the top of the screen to search for the storage account you created earlier. Click on the correct result to load the storage account's overview pane.
1. In the "Services" section, you will links labeled "Blobs" and "Files" inside of square bounding boxes. Click on "Blobs".
1. A list of blob containers will be displayed. Click on the container named "blobfuse" created by this tutorial.
1. A navigable directory structure will be displayed. You will find the output images from the evaluation job under the `evaluation_output` folder. You can download files by clicking on the filename and then clicking "Download" in the pane that appears at right.

These extracted TIF images will each be ~12 MB in size and can be examined using a web browser (and most image/photo editing software).

### Review of sample results

Full-sized sample output images are provided in the [outputs folder](./outputs) of this repository; we provide a side-by-side, zoomed-out view below for easy comparison. While the tutorial creates a model trained for only one epoch, we also include an illustration of the model's output after 250 epochs to illustrate the accuracy that can ultimately be achieved:

<img src="./outputs/comparison_fullsize.PNG"/>

You'll notice that the model trained for 250 epochs has predicted the presence of a body of water (near top-center). This body of water is indeed present, demonstrating the potential of a sufficiently-trained model to suggest improvements on our ground-truth labels.

## Next steps

When you are done, we recommend deleting all resources you created for this tutorial. Please see the instructions in the [setup section](./setup.md) of this tutorial.
