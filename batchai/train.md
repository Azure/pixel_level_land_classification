# Train a pixel-level land classification model

In this section, you will use your Batch AI cluster to train an image segmentation (pixel-level classification) model using sample aerial imagery and ground truth labels. This section assumes that you have already completed the [setup steps](./setup.md) described previously and have downloaded the files provided in [this git repository](https://github.com/Azure/pixel_level_land_classification). When you have completed this section, we recommend that you test its performance by [applying your model to new aerial images](./evaluate.md).

## Submit the training job

Before executing the command below, ensure that the `training_job.json` file contained in this git repository has been downloaded to your computer and is available on the path.
```
az batchai job create -n trainingjob -f training_job.json -c batchaidemo --resource-group %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
```

This job will take 10-20 minutes to run; while waiting, you can read the section below for more information on what the job is doing. You can also check on the job's progress using the following command:
```
az batchai job show -n trainingjob --resource-group %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
```

When the job status indicated by "executionState" changes from "running" to "succeeded", the training job is complete. You can also monitor the standard output and error messages as they're produced using the following commands:
```
az batchai job file stream -d stdouterr -j trainingjob -f stdout.txt -g %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
az batchai job file stream -d stdouterr -j trainingjob -f stderr.txt -g %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
```

To exit the streaming view, press Ctrl+C. You will be asked whether to terminate the job if it is still running: press "N" to indicate that you want the job to continue running.

## Understand the training job

### The job config file, `training_job.json`

The `training_job.json` file specifies where the training script is located and what arguments it will take, as well as how the distributed training job should be launched. Since we've specified that this is a CNTK training job, Batch AI will launch the job using `mpiexec` to coordinate distributed training between the specified number of workers/processes. (Batch AI also streamlines distributed training for [other supported deep learning frameworks](https://github.com/Azure/BatchAI/tree/master/recipes).) Notice that filepaths in `training_job.json` are defined relative to `$AZ_BATCHAI_MOUNT_ROOT`, the location on each of your cluster's VMs where the file share (`$AZ_BATCHAI_MOUNT_ROOT/afs`) and blob storage container (`$AZ_BATCHAI_MOUNT_ROOT/blobfuse`) have been mounted.


By default, the model will be trained for just one epoch (see the `num_epochs` parameter in `training_job.json`) and the eight provided training images pairs. This choice minimizes the runtime of the tutorial but will not result in a very performant model. For comparison, our full-scale training was performed for 250 epochs using 740 training image pairs.

### Training data access

Near the beginning of the [training script](https://ai4ehackathons.blob.core.windows.net/landcovertutorial/scripts/train_distributed.py) is a custom minibatch source specifying how the training data should be read and used. Our training data comprise pairs of TIF images. The first image in each pair is a four-channel (red, green, blue, near-infrared) aerial image of a region of the Chesapeake Bay watershed. The second image is a single-channel "image" corresponding to the same region, in which each pixel's value corresponds to a land cover label:
- 0: Unknown land type
- 1: Water
- 2: Trees and shrubs
- 3: Herbaceous vegetation
- 4+: Barren and impervious (roads, buildings, etc.); we lump these labels together

These two images in each pair correspond to the features and labels of the data, respectively. The minibatch source specifies that the available image pairs should be partitioned evenly between the workers, and each worker should load its set of image pairs into memory at the beginning of training. This ensures that the slow process of reading the input images is performed only once per training job. To produce each minibatch, subregions of a given image pair are sampled randomly. Training proceeds by cycling through the image pairs.

### The model architecture
The [model definition script](https://ai4ehackathons.blob.core.windows.net/landcovertutorial/scripts/model_mini_pub.py) specifies the model architecture: a form of [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/). The input for this model will be a 256 pixel x 256 pixel four-channel aerial image (corresponding to a 256 meter x 256 meter region), and the output will be predicted land cover labels for the 128 m x 128 m region at the center of the input region. (Predictions are not provided at the boundaries due to edge effects.)

### Training script outputs

We encourage the reader to explore the output files that the training script writes to disk. While the script is still running, you will find that standard output and error messages are written to disk in the Azure File Share, which you can access as follows:
1. Log into the [Azure Portal](https://portal.azure.com).
1. Use the search bar along the top of the screen to search for the storage account you created earlier. Click on the correct result to load the storage account's overview pane.
1. In the "Services" section, you will links labeled "Blobs" and "Files" inside of square bounding boxes. Click on "Files".
1. A list of file shares in the storage account will appear. Click on the file share named "batchai" created by this tutorial.
1. A navigable directory structure will now be displayed. Explore this file system to find the logs:
    - The logs created during cluster creation are under `batchai/[subscription id]/[storage account name]/clusters/batchaidemo`.
    - The logs created during training are under `batchai/[subscription id]/[storage account name]/jobs/trainingjob`.

   Note that you can download a file by clicking on its name and choosing the "Download" option on the pane that appears along the right side of your screen.

When training has completed, the trained model will be written to blob storage. You can find this file from Azure Portal as follows:
1. Navigate to the storage account's overview pane as described above, then click on "Blobs".
1. A list of blob containers will be displayed. Click on the container named "blobfuse" created by this tutorial.
1. A navigable directory structure will be displayed. You may find the following folders:
    - `scripts`: Contains the training and evaluation scripts.
    - `training_data` and `evaluation_data`: Contain the TIF files used for training and evaluation.
    - `models`: Contains the model files created during training.
    - `evaluation_output` (to be created later by the evaluation script): Contains the TIF files that illustrate the model's output labels for a sample region.

    Note that you can download a file by clicking on its name and choosing the "Download" option in the pane that appears at right.

In your future work, you may find it handy to access these files without navigating through the Portal. Additional options include:
- [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) for Windows/Linux/Mac - provides an alternative GUI interface
- Mounting an Azure File Share as a local disk ([Windows](https://docs.microsoft.com/en-us/azure/storage/files/storage-how-to-use-files-windows)/[Linux](https://docs.microsoft.com/en-us/azure/storage/files/storage-how-to-use-files-linux)/[Mac](https://docs.microsoft.com/en-us/azure/storage/files/storage-how-to-use-files-mac))
- Using the Azure CLI to [retrieve files from Azure storage accounts](https://docs.microsoft.com/en-us/azure/storage/common/storage-azure-cli) (see also the [API doc](https://docs.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest))
- Connecting to a node in your cluster via SSH or SCP and accessing the storage that has been mounted there (typically under `/mnt/batch/tasks/shared/LS_root/mounts`)
- Retrieving job output with a [Batch AI SDK](https://github.com/Azure/BatchAI) (available for Python, C#, Java, and Node.js)

## Next steps

Now that you have produced a trained model, you can test its performance in the following section on [applying your model to new aerial images](./evaluate.md). You may later wish to return to this section and train a model for more than one epoch, to improve its performance.

You may like to learn more about [scaling our training method](./scaling.md) for larger datasets and clusters.

You may also be interested in using your trained model in ArcGIS Pro. Click [here](../README.MD) to return to the main page of this repository, where you can find directions on how to provision a Geo AI DSVM with ArcGIS Pro installed, and employ your trained model in an ArcGIS project.

If you decide to stop pursuing the tutorial after this step, we recommend deleting all Azure resources you created. Please see the instructions in the [setup section](./setup.md) of this tutorial.
