# Pre-requisites for MLADS Lab: AI for Earth pixel-level land classification with CNTK, the Geo AI DSVM, and Batch AI

Attendees are welcome whether or not they complete some/all of the hands-on components in this tutorial. The "pre-requisites" below are only necessary if you intend to create the Azure resources needed for the hands-on components of the tutorial. The total cost of these resources is estimated to be ~$10/hr: to save money, we recommend that you test provisioning in your subscription well in advance and swiftly delete these test resources, then recreate them again shortly before the tutorial starts.

If you hoped to participate in the hands-on components of the tutorial, but are unable to create these resources for technical or financial reasons, please email Mary Wahl (mawah@microsoft.com): our team may be able to assist.

1) Ensure that your Azure subscription has sufficient quota for NC series virtual machines

	If you do not already have a paid Azure subscription, you will need to [create one](https://azure.microsoft.com/en-us/pricing/purchase-options/). (Note that free Azure subscriptions have restrictions in place that prevent them for creating the resources needed in this tutorial.)

	To check your NC series quota, log into [Azure Portal](http://portal.azure.com), click on "Subscriptions" in the left-hand menu, and then click on the name of the subscription you'd like to use. Click on "Usage + quotas" in the left-hand menu of the subscription's pane. Search for "Standard NC Family vCPUs" and ensure that you have eighteen cores (enough for three NC6 VMs) in at least one region. Note the name of this region, as you'll use it when creating VMs and Batch AI clusters.

	If you find you do not have sufficient quota, you will need to submit two support tickets: the first will increase your NC series core quota in general, and the second will increase the NC series core quota for Batch AI specifically. Click on the help icon (circled question mark) along the top of Azure Portal, then click "Help + Support" in the drop-down menu. In the new pane that appears, click on "New support request" in the left-hand menu. Under quota type, choose "Compute (cores/vCPUs)" or "Batch AI", as appropriate. Follow the remaining prompts to complete your support ticket, being sure to specify the region (e.g. East US) and total number of cores you will need (which may be greater than 18, if you already own other NC series VMs in that region).

2) Provision an NC6 (single-GPU) Geo AI Data Science Virtual Machine and connect to it via remote desktop

	Follow [these instructions](https://github.com/Azure/pixel_level_land_classification/blob/master/geoaidsvm/setup.md) to provision your Geo AI Data Science Virtual Machine and access it via Remote Desktop.

3) Install the Azure CLI on the laptop you'll use at the tutorial

	Follow [these instructions](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) to install the Azure CLI on your laptop.

4) Create a Batch AI cluster with two NC6 VMs

	Follow [these instructions](https://github.com/Azure/pixel_level_land_classification/blob/master/batchai/setup.md) to create a Batch AI cluster using the Azure CLI on your laptop. You may find that you need to increase your Batch AI quota for NC series VMs; if so, follow the instructions under item (1) above.
  
  We look forward to seeing you at the tutorial!
