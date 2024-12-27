locals {
  # template labels for all resources
  common_labels = {
    "country" = var.country
    "environment" = var.environment
    "application-id" = var.application_id
  }
}
