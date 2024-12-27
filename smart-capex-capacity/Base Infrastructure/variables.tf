## General
variable "country" {
  type        = string
  default     = ""
  description = "Dountry id"
}
variable application_id {
  type        = string
  default     = ""
  description = "application_id"
}
variable "project_id" {
  type        = string
  description = "The id of the GCP project."
}

variable "region" {
  type        = string
  description = "The region of the GCP project."
}
variable "environment" {
  type        = string
  default     = ""
  description = "Environnement"
}
variable "workload" {
  type        = string
  default     = ""
  description = "The workload for which the ressources is created"
}
variable "python_artifact_registry_purpose" {
  type        = string
  default     = ""
  description = "the purpose which will be used to build the artifact registry name"
}
variable "docker_artifact_registry_purpose" {
  type        = string
  default     = ""
  description = "the purpose which will be used to build the artifact registry name"
}



## Cloud Storage
# Buckets
variable "buckets" {
  type        = set(map(string))
  default = []
  description = "The list of Buckets to create on project: list of object with purpose, type and versioning properties"
}

## BQ
# BQ Dataset
variable "datasets" {
  type        = set(map(string))
  default = []
  description = "The list of Dataset to create on project: list of object with purpose and description properties"
}

variable "repositories" {
  type        = set(map(string))
  default = []
  description = "The list of artifact repositories to create on project"
}

##############################
##         GCP APIs         ##
##############################
variable "project_services" {
  type = list(string)
  default = [
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com",
    "storage-api.googleapis.com",
    "bigquery.googleapis.com",
    "bigquerystorage.googleapis.com",
    "run.googleapis.com",
    "ml.googleapis.com",
    "iam.googleapis.com",
    "eventarc.googleapis.com",
  ]
  description = "Set of services to enable on the project"
}