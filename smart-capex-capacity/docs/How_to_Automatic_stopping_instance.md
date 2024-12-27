# Automatic Stopping of Notebook Instance using Cloud Scheduler

This README will guide you through the process of setting up a Google Cloud Scheduler job to automatically stop a Google Cloud Compute Engine instance, including Notebooks, by making a POST request to the Compute Engine REST API.

## Prerequisites
- A Google Cloud Platform account.
- Basic knowledge of Google Cloud Console.
- A running Google Cloud Compute Engine instance.

## Instructions

### 1. Enable necessary APIs
Ensure the following APIs are enabled:
- Cloud Scheduler API
- Compute Engine API

You can enable them by visiting the GCP Console, navigate to APIs & Services -> Library, search for the above APIs, and enable each one of them.

### 2. Create a Service Account
We need a service account with the necessary permissions to stop the instance.
- Navigate to IAM & Admin -> Service accounts.
- Click on `Create Service Account`.
- Provide a name for the service account e.g. `stop-instance-service-account`.
- For roles, assign `Compute Instance Admin (v1)` to allow the service account to control Compute Engine instances.
- Create a key for this service account and download it in JSON format. We will need this to authenticate our Cloud Scheduler job.

### 3. Create a Cloud Scheduler Job
This job will make a POST request to the Compute Engine API at specified intervals to stop the instance.
- Go to GCP Console -> Cloud Scheduler -> Create Job.
- For Frequency, provide a cron schedule e.g. `0 23 * * *` to run the job every day at 11 PM.
- For Target, select `HTTP`.
- For URL, provide the stop endpoint URL for your instance: `https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instances/{resourceId}/stop`, replacing `{project}`, `{zone}`, and `{resourceId}` with your project ID, zone, and instance ID respectively.
- For HTTP method, select `POST`.
- In the `Show more` section:
  - For Auth header, select `Add OIDC token`.
  - For Service account, select the service account created in step 2.
  - For Audience, provide the same URL as above.

## Conclusion
You have successfully created a setup where a Cloud Scheduler job will automatically stop your Compute Engine instance at a specified time. 

Please replace `{project}`, `{zone}`, and `{resourceId}` with your actual project ID, zone, and instance ID.

Also, bear in mind that there might be additional charges related to the use of Cloud Scheduler. Always ensure to check the pricing details before proceeding with the setup.

Remember to keep the service account key safe. If you lose it or it gets compromised, delete the service account and create a new one.

## Disclaimer
This setup provides no guarantees of uptime or data preservation. It's intended to help manage costs by shutting down instances when they're not needed, but it should be used with caution in production environments. Always have a disaster recovery plan in place.
