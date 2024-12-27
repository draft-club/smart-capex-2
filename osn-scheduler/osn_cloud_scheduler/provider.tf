provider "google" {
   credentials = "${file("./creds/serviceaccount.json")}"
   project     = "oro-smart-capex-001-dev" # REPLACE WITH YOUR PROJECT ID
   region      = "europe-west3"
 }
 #credentials = file("${path.module}/osn_cloud_scheduler/creds/serviceaccount.json")