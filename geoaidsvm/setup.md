# Provision and access a Geo AI Data Science VM

## Provision the Geo AI Data Science VM

We recommend that you provision a Geo AI Data Science VM using the [one-click provisioning solution template](https://gallery.azure.ai/Solution/Geo-AI-Data-Science-Virtual-Machine-2). However, if you prefer, you can create the VM from the Azure Portal using the instructions below:

1. Log into the [Azure Portal](https://portal.azure.com) with your preferred Azure account.
1. Click the "Create a resource" button at top left.
1. In the marketplace search bar, type "Geo AI Data Science VM with ArcGIS" and press Enter. (Auto-completion will not work until the VM is publicly available.)
1. Click on the corresponding result, then click "Create" to begin customizing your VM.
1. In the Basics pane, provide your desired credentials, subscription, resource group, and deployment region, then click "OK."
    - We recommend creating a new resource group, so that all associated resources can be easily deleted when you are done with the tutorial.
    - If you have no preference, we recommend using the "East US" location: this will speed up transfer of tutorial files to your machine.
1. On the Settings pane, choose an "NC6," "NC12," or "NC24" VM size to select a VM with 1, 2, or 4 GPUs, then click "OK".
    - The tutorial can be completed with a single GPU. If you would like to explore the time savings achieved by training with multiple GPUs, you will need to select either an NC12 or NC24 VM size.
    - If you do not see the "NC" VM sizes as options, you may need to choose a different location on the Basics pane.
    - If you do not choose an "NC" VM size, you will likely encounter failures during model training later in this tutorial.
1. Confirm your choices and buy the VM to begin deployment.

Deployment will likely take 10-20 minutes to complete. When it has finished, a link to your VM should be added to your dashboard in Azure Portal. (You can also find your VM by name using the search utility in Azure Portal.)

## Access the Geo AI Data Science VM

1. Navigate to your VM's pane in Azure Portal.
1. Along the top of the pane, you should see a "Connect" button. Click this to download an RDP file for connection to your VM via remote desktop.
1. Your connection instructions will vary depending on your local machine's operating system:
    - On Windows, double-click on the RDP file and supply the credentials you chose during provisioning to connect. (You may need to choose "Use a different account" and place a backslash in front of your username to override your default domain settings.)
    - On Mac, you will need to install an application to connect using the RDP file. Further instructions are available [here](https://docs.microsoft.com/en-us/windows-server/remote/remote-desktop-services/clients/remote-desktop-mac).
1. Once you have connected to your VM, click on the "Jupyter" icon on the desktop to launch a Jupyter Notebook server. 
    - This will launch a new browser window or tab showing the Notebook Dashboard, a control panel that allows you to select which notebook to open.
    - You may need to copy a URL from the auto-launched window into your preferred web browser in order to view the provided sample notebooks.
1. Navigate to the "notebooks\GeoBooks\AI-for-Earth" subfolder and click on the `01_Intro_to_pixel-level_land_classification.ipynb` file to launch it.

This notebook will contain an explanation of the other notebooks in the subfolder, including a recommended order for reading and executing them. Note that code cells within the notebook can be executed by selecting the cell and pressing Ctrl+Enter.
