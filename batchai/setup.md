# Prerequisites and setup steps

In this section of the tutorial on [pixel-level land classification from aerial imagery](https://github.com/Azure/pixel_level_land_classification), we describe the steps needed to create an Azure Batch AI cluster with access to all necessary files to complete this tutorial. Once you have completed this section, you'll be ready to [train a model from scratch](./train.md) on a GPU cluster (for distributed training at scale) using our sample data and provided scripts.

## Prerequisites

**Azure Subscription**

This tutorial will require an [Azure subscription](https://azure.microsoft.com/en-us/free/) with sufficient quota to create a storage account and two NC6 (single-GPU) VMs as a Batch AI cluster. This tutorial will likely take two hours to complete on the first pass.

**Files from this repository**

You will need local copies of the .json files included in this git repository. We recommend that you download or clone the full repository locally, but you can also download each file individually. (If you choose that approach, be careful to download the "raw" files -- it's common to accidentally save GitHub's HTML previews of the files instead.)

**Utilities**

This tutorial requires the following programs:
- [Azure CLI 2.0](https://docs.microsoft.com/cli/azure/install-azure-cli)
- [AzCopy](https://docs.microsoft.com/azure/storage/common/storage-use-azcopy)

These programs are available for Windows and Linux. If you prefer not to install these programs locally, you may instead provision an [Azure Windows Data Science Virtual Machine](https://docs.microsoft.com/azure/machine-learning/data-science-virtual-machine/provision-vm). (Both programs are pre-installed on this VM type and are available on the system path.) The commands included in this tutorial were written and tested in Windows, but readers will likely find it straightforward to adapt for Linux.

Once these programs are installed, open a command line interface and check that the binaries are available on the system path by issuing the commands below:
```
az
azcopy
```
If not, you may need to [edit the system path](http://www.zdnet.com/article/windows-10-tip-point-and-click-to-edit-the-system-path-variable/) to point to the folders containing these binaries (e.g., `C:\Program Files (x86)\Microsoft SDKs\Azure\AzCopy`) and load a fresh command prompt.

This tutorial was tested using Azure CLI version 2.0.42. If you installed the Azure CLI previously, check your version number using `az --version` and upgrade if necessary.

### Prepare to use the Azure CLI

In your command line interface, execute the following command. The output will contain a URL and token that you must visit to authenticate your login.
```
az login
```

You will now indicate which Azure subscription should be charged for the resources you create in this tutorial. List all Azure subscriptions associated with your account:
```
az account list
```

Identify the subscription of interest in the JSON-formatted output. Use its "id" value to replace the bracketed expression in the command below, then issue the command to set the current subscription.
```
az account set -s [subscription id]
```

## Create the necessary Azure resources

### Create an Azure resource group

We will create all resources for this tutorial in a single resource group, so that you may easily delete them when finished. Choose a name for your resource group and insert it in place of the bracketed expression below, then issue the commands:
```
set AZURE_RESOURCE_GROUP=[resource group name]
az group create --name %AZURE_RESOURCE_GROUP% --location eastus
```
You may use other locations, but we recommend `eastus` for proximity to the data that will be copied into your storage account, and because the necessary VM type (NC series) is available in the East US region.

### Create an Azure storage account and populate it with files

We will create an Azure storage account to hold training and evaluation data, scripts, and output files. Choose a unique name for this storage account and insert it in place of the bracketed expression below. Then, issue the following commands to create your storage account and store its randomly-assigned access key:
```
set STORAGE_ACCOUNT_NAME=[storage account name]
az storage account create --name %STORAGE_ACCOUNT_NAME% --sku Standard_LRS --resource-group %AZURE_RESOURCE_GROUP% --location eastus
for /f "delims=" %a in ('az storage account keys list --account-name %STORAGE_ACCOUNT_NAME% --resource-group %AZURE_RESOURCE_GROUP% --query "[0].value"') do @set STORAGE_ACCOUNT_KEY=%a
```

With the commands below, we will create an Azure File Share to hold setup and job-specific logs, as well as an Azure Blob container for fast file I/O during model training and evaluation. (The file share offers more options for retrieving your log files, while data access will be faster from blob containers.) Then, we'll use AzCopy to copy the necessary data files for this tutorial to your own storage account.  Note that we will copy over only a subset of the available data, to save time and resources.
```
az storage share create --account-name %STORAGE_ACCOUNT_NAME% --name batchai
az storage container create --account-name %STORAGE_ACCOUNT_NAME% --name blobfuse
AzCopy /Source:https://ai4ehackathons.blob.core.windows.net/landcovertutorial /SourceSAS:"https://ai4ehackathons.blob.core.windows.net/landcovertutorial?se=2020-04-06T06%3A59%3A00Z&sp=rl&sv=2018-03-28&sr=c&sig=YD6mbqnmYTW%2Bs6guVndjQSQ8NUcV8F9HY%2BhPNWiulIo%3D" /Dest:https://%STORAGE_ACCOUNT_NAME%.blob.core.windows.net/blobfuse /DestKey:%STORAGE_ACCOUNT_KEY% /S
```

Expect the copy step to take 5-10 minutes.

### Create an Azure Batch AI workspace and experiment

Batch AI workspaces can contain clusters as well as experiments, which in turn organize jobs. Choose a unique name for your workspace and experiment and insert them in place of the bracketed expressions below, then run the commands to create a workspace and experiment:

```
set WORKSPACE_NAME=[your selected workspace name]
az batchai workspace create -n %WORKSPACE_NAME% --resource-group %AZURE_RESOURCE_GROUP%

set EXPERIMENT_NAME=[your selected experiment name]
az batchai experiment create -n %EXPERIMENT_NAME% -w %WORKSPACE_NAME% --resource-group %AZURE_RESOURCE_GROUP% 
```

### Create an Azure Batch AI cluster

We will create an Azure Batch AI cluster containing two NC6 Ubuntu DSVMs. This two-GPU cluster will be used to train our model and then apply it to previously-unseen data. Before executing the command below, ensure that the `cluster.json` file provided in this repository (which specifies the Python packages that should be installed during setup) has been downloaded to your computer and is available on the path (you may need to change directories to the `batchai` folder of your cloned copy of this repository). We also recommend that you change the username and password to credentials of your choice.
```
az batchai cluster create -n batchaidemo --user-name lcuser --password lcpassword --afs-name batchai --image UbuntuDSVM --vm-size STANDARD_NC6 --max 2 --min 2 --storage-account-name %STORAGE_ACCOUNT_NAME% --bfs-name blobfuse --bfs-mount-path blobfuse -f cluster.json --resource-group %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% 
```
This command will create a cluster whose credentials are a username-password pair. For increased security, we highly encourage the use of an SSH key as credential: for more information, see the [Batch AI documentation](https://github.com/Azure/BatchAI/blob/master/documentation/using-azure-cli-20.md#Admin-User-Account) and the output of the `az batchai cluster create -h` command.

It will take approximately ten minutes for cluster creation to complete. You can check on progress of the provisioning process using the command below: when provisioning is complete, you should see that the "errors" field is null and that your cluster has two "idle" nodes.
```
az batchai cluster show -n batchaidemo --resource-group %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME%
```

## Next steps

You have now completed all of the setup steps required for this tutorial. We recommend proceeding to the [model training](./train.md) section of this repository.

Click [here](../README.MD) to return to the main page of this repository for more information.

## Cleanup

When you have completed all sections of interest to you in this repository, be sure to delete the resources you created with the following command:
```
az group delete -n %AZURE_RESOURCE_GROUP% 
```
