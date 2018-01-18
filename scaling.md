# Scaling up training

If you have completed our [tutorial on pixel-level classification](https://github.com/Azure/pixel_level_land_classification), you've trained an image segmentation model for one epoch using two GPUs and eight training image pairs in <20 minutes. For our work with the Chesapeake Conservancy, we trained a similar model for 250 epochs using 148 GPUs and 1,111 training image pairs -- all in a little over two hours. This section describes how this scaling was achieved and considerations the reader might apply to their own projects.

## Increasing worker (GPU) count

### Decreasing epoch length

Doubling the number of workers would ideally decrease the time required to train the model by half. In practice, the actual speed-up is lower due to overhead in communication between workers and our use of a synchronous training method. We show below that we achieved near-linear speed-ups between 1 and 64 workers, improvements in training time eventually becoming more marginal. (We report the average epoch duration after training data load to memory, a time-intensive step which does not scale with worker number.)

<img src="outputs/epoch_duration_scaling.png">

We expect that the following modifications would further improve training time for large clusters (though we did not pursue them for this use case):
- Using workers connected by Infiniband (e.g. the [NC24r Azure VM SKU](https://azure.microsoft.com/en-us/blog/azure-n-series-general-availability-on-december-1/)) to speed up communication between workers
- Using [1-bit Stochastic Gradient Descent](https://docs.microsoft.com/en-us/cognitive-toolkit/enabling-1bit-sgd) to decrease the size of messages passed between workers
- Using [blockwise model update and filtering, aka block momentum](https://docs.microsoft.com/en-us/cognitive-toolkit/Multiple-GPUs-and-machines#6-block-momentum-sgd) during training to decrease the frequency of communication between workers
- Tuning minibatch size to decrease frequency of communication between workers
- Pursuing an asynchronous training approach

### Permitting data load to memory

Increasing worker count is also beneficial when it permits the dataset to be stored entirely in memory. Accessing data from a remote store, or even from disk, can be rate-limiting for training, so it is ideal for each worker to perform an initial data load and then access data from memory in subsequent rounds of training. This becomes achievable using data-parallel training when the number of workers is sufficiently large.

### How to implement

To increase the number of worker nodes in your cluster during deployment, simply modify the "targetNodeCount" and "vmSize" values in the `cluster.json` file. We recommend that you use a [VM SKU](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/overview#vm-sizes) with a larger number of GPUs where possible, e.g. create a cluster with four NC24 VMs rather than a cluster with sixteen NC6 VMs. This option will reduce the average communication time between workers and will not impact the average memory/CPU/storage per worker.

## Data access

### Network File System

At the time that we performed full model training for the Chesapeake Conservancy (9/2017), Azure Batch AI did not yet offer data access from blob storage via blobfuse. Instead, we provisioned a Network File System (NFS) to host our data for concurrent access by many workers. This option is preferable to storing data on an Azure File Share, but we believe that accessing data from blob storage (as demonstrated in this tutorial) will now be preferable for most users.

If you would like to try using an NFS as your data store, you may modify the setup steps in [setup.md](./setup.md) to create a file server and mount it on a new cluster. You can use your favorite SSH or SCP agent to upload your data files under the `/mnt/data` directory of the file server, so that they will be accessible from your cluster.
```
az batchai file-server create -n batchaidemo -u yourusername -p yourpassword --vm-size Standard_D2_V2 --disk-count 1 --disk-size 1000 --storage-sku Standard_LRS
for /f "delims=" %a in ('az batchai file-server list -g %AZURE_RESOURCE_GROUP% --query "[?name == 'batchaidemo'].mountSettings.fileServerPublicIp | [0]"') do @set AZURE_BATCH_AI_TRAINING_NFS_IP=%a
echo %AZURE_BATCH_AI_TRAINING_NFS_IP%
az batchai cluster create -n batchaidemo -u lcuser -p lcpassword --afs-name batchai --nfs batchaidemo --image UbuntuDSVM --vm-size STANDARD_NC6 --max 2 --min 2 --storage-account-name %STORAGE_ACCOUNT_NAME% --container-name blobfuse --container-mount-path blobfuse -c cluster.json
```

You may also wish to use a premium storage SKU (learn more from the output of `az batchai file-server create -h`) or [another VM SKU](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/overview#vm-sizes) to improve the 
