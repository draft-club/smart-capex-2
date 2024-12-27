

#resource "google_vpc_access_connector" "connector" {
#  name          = "vpc-connector"
#  region        = var.region
#  project       = var.project_id
#  ip_cidr_range = "10.12.0.0/28"
#  network       = "default"
#}




#data "google_vpc_access_connector" "existing_connector" {
#  name    = "vpc-connector"
#  region  = var.region
#  project = var.project_id
#}



resource "google_vpc_access_connector" "existing_connector" {
#project = var.project_id
project = "oro-network-shared-054-prd"
name = "vpc-connector"
#provider = google-beta
region = var.region
ip_cidr_range = "172.17.183.112/28"
network = "oro-xpn-network-prod"
machine_type = "e2-micro"
min_instances = 2
max_instances = 5
max_throughput = 500
depends_on = [google_project_service.vpcaccess_api]
}



resource "google_cloud_run_service" "default" {
  name     = "angular-app-vtf"
  location = var.region
  #project  = var.project_id
  project = "oro-smart-capex-001-dev"

  metadata {
    annotations = {
      "run.googleapis.com/ingress" : "internal-and-cloud-load-balancing"
    }
  }
  template {
    metadata {
      annotations = {
        #"run.googleapis.com/vpc-access-connector" : google_vpc_access_connector.connector.name
        "run.googleapis.com/vpc-access-connector" : data.google_vpc_access_connector.existing_connector.name
      }
    }
    spec {
      containers {
        ports {
          container_port = 80
        }
        image = "europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-dataquality/ocdvt@sha256:81ff5181f9e0a29d4b5e82ceb090bb2eadf827bbd618dd382fb2b976be59a978"
      }
    }
  }
} 

resource "google_compute_region_network_endpoint_group" "serverless_neg" {
  lifecycle {
    create_before_destroy = true
  }

  provider              = google
  name                  = "serverless-neg1"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  cloud_run {
    service = google_cloud_run_service.default.name
  }
}

module "lb-http" {
  source  = "GoogleCloudPlatform/lb-http/google//modules/serverless_negs"
  version = "5.1.0"

  #project = var.project_id
  project = "oro-smart-capex-001-dev"

  name    = var.lb_name

  ssl                             = true
  managed_ssl_certificate_domains = [var.domain]
  https_redirect                  = true

  backends = {
    default = {
      description = null
      groups = [
        {
          group = google_compute_region_network_endpoint_group.serverless_neg.id
        }
      ]
      enable_cdn             = false
      security_policy        = null
      custom_request_headers = null

      iap_config = {
        enable               = true
        oauth2_client_id     = var.iap_client_id
        oauth2_client_secret = var.iap_client_secret
      }
      log_config = {
        enable      = false
        sample_rate = null
      }
    }
  }
}

data "google_iam_policy" "iap" {
  binding {
    role = "roles/iap.httpsResourceAccessor"
    members = [
      #"group:everyone@google.com", // a google group
      "allAuthenticatedUsers"          // anyone with a Google account (not recommended)
      // "user:jaleleddine.hajlaoui@orange.com", // a particular user
    ]
  }
}

resource "google_iap_web_backend_service_iam_policy" "policy" {
  #project             = var.project_id
  project = "oro-smart-capex-001-dev"
  web_backend_service = "${var.lb_name}-backend-default"
  policy_data         = data.google_iam_policy.iap.policy_data
  depends_on = [
    module.lb-http
  ]
}

output "load-balancer-ip" {
  value = module.lb-http.external_ip
}

output "oauth2-redirect-uri" {
  value = "https://iap.googleapis.com/v1/oauth/clientIds/${var.iap_client_id}:handleRedirect"
}

