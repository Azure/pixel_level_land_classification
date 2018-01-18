# Apply the trained model to new data

We will apply the trained model to an approximately 1 km x 1 km region centered at a point in Charlotte County, VA. This region is close to (but does not overlap with) the regions used to train the model, so we expect reasonable performance despite the short training time.

## Submit the evaluation job

We will launch an evaluation job to apply our trained model to a specified region in our evaluation data. Before executing the command below, ensure that the `evaluation_job.json` file contained in this git repository has been downloaded to your computer and is available on the path, and that the training job has finished running.
```
az batchai job create -n evaluationjob -c evaluation_job.json -r batchaidemo
```
This job will take ~5 minutes to run; while waiting, you can read the section below for more information on what the job is doing. You can check on the job's progress using the following command:
```
az batchai job show -n evaluationjob
```

## Understand the output of the evaluation job

### Description of what the evaluation job is doing

### Accessing your evaluation job's output

### Sample results

<img src="./outputs/comparison_fullsize.PNG"/>

## Next steps

When you are done, we recommend deleting all resources you created for this tutorial. Please see the instructions in the [setup section](./setup.md) of this tutorial.
