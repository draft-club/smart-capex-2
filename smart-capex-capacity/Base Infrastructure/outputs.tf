# BQ Datsets
output "datasets_list" {
  value = {
        for ds in google_bigquery_dataset.dataset : 
            ds.dataset_id => {"type": "BQ Dataset", "name": ds.dataset_id, "description": ds.description, "location": ds.location}
        }
  ##ok
#   value = values(google_bigquery_dataset.dataset).*.dataset_id
}

# Cloud Storage buckets 
output "buckets_list" {
  value = {
        for b in google_storage_bucket.buckets: 
            b.name => {"type": "GCS bucket","name": b.name, "location": b.location}
        }
}

# Artifact Registry repositories 
output "repositories_list" {
  value = {
        for r in google_artifact_registry_repository.repositories: 
            r.name => {"type": "Artifact Registry repository","name": r.repository_id, "location": r.location, "format": r.format, "description": r.description}
        }
}

# Service Accounts
# output "features_sa" {
#   value = { 
#     "account_id": google_service_account.features-sa.account_id,
#     "description": google_service_account.features-sa.display_name
#     }
# }

# # Monitoring Notification Channel
#output "notification_channel_list" {
#  value = {
#        for r in google_monitoring_notification_channel.notificationchannel: 
#            r.name => {"type": "Artifact Registry repository","name": r.display_name, "type": r.type, "labels": replace(replace(jsonencode(r.labels), "\"", ""), ":", "="), "description": r.description}
#        }
#}
