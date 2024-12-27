variables:
  GCP_SERVICE_ACCOUNT: "sa-oro-data-cicd-capex-dev@oro-smart-capex-001-dev.iam.gserviceaccount.com"
  PROJECT_ID: "oro-smart-capex-001-dev"
  REGION: "europe-west3"
  REPOSITORY: "smart-capex-capacity/smartcapex-churn"
  IMAGE: 'churn'
  IMAGE_TAG: 'churn:latest'

before_script:
  - echo "Authenticating with Google Cloud..."
  - echo ${CI_JOB_JWT_V2} > .ci_job_jwt_file
  - gcloud config set project $PROJECT_ID
  - gcloud iam workload-identity-pools create-cred-config "${GCP_WORKLOAD_IDENTITY_PROVIDER}" \
      --service-account="${GCP_SERVICE_ACCOUNT}" \
      --output-file=.gcp_temp_cred.json \
      --credential-source-file=.ci_job_jwt_file
  - gcloud auth login --cred-file=.gcp_temp_cred.json

deploy:
  image: google/cloud-sdk:slim
  stage: deploy
  script:
    # Configure Docker with Artifact Registry
    - gcloud auth configure-docker $REGION-docker.pkg.dev

    # Check if the repository exists, and create it if not
    - |
      check_for_repo=$(gcloud artifacts repositories describe $REPOSITORY --location=$REGION 2>&1 >/dev/null)
      if [[ $check_for_repo == ERROR* ]]; then
        echo "Creating a repository named $REPOSITORY"
        gcloud artifacts repositories create $REPOSITORY \
          --repository-format=docker \
          --location=$REGION \
          --description="Vertex AI Training Custom Containers"
      else
        echo "There is already a repository named $REPOSITORY"
      fi

    # Build and push the Docker image to Artifact Registry
    - docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG .
    - docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG

  artifacts:
    paths:
      - .ci_job_jwt_file
      - .gcp_temp_cred.json
