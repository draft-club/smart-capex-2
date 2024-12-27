/*
resource "google_storage_bucket" "auto-expire" {
  name          = "cloudquicklabs_gcp_bucket_iac1"
  location      = "europe-west3" # Use a valid location
  force_destroy = true

  public_access_prevention = "enforced"
  uniform_bucket_level_access = true
}
*/



resource "google_cloud_scheduler_job" "job" {
  name             = "temp_test_tf"
  description      = "test http job"
  schedule         = "0 0 * * 1"
  time_zone        = "Europe/Paris"
  attempt_deadline = "320s"

  retry_config {
    retry_count = 1
  }

http_target {
    http_method = "POST"
    uri         = "https://ocdvt-goykkb7jhq-ew.a.run.app/validation"
    
 
    
    body        = base64encode(jsonencode({
      "mode"                     : "bq",
      "file_type"                : "oss_counters_3g_huawei",
      "project_id"             : "oro-smart-capex-001-dev",
      "bigquery_dataset_name"    : "smart_capex_raw",
      "table_name"               : "oss_counters_3g_huawei",
      "delimiter"                : ",",
      "bucket_name"              : "oro-smart-capex-config-001-dev"
    }))
headers = {
      "Content-Type" = "application/json"
      "User-Agent"   = "Google-Cloud-Scheduler"
    }



    oidc_token {
      service_account_email = "sa-oro-data-cicd-capex-dev@oro-smart-capex-001-dev.iam.gserviceaccount.com"
      audience              = "https://ocdvt-goykkb7jhq-ew.a.run.app/validation"
    }
  }
}

