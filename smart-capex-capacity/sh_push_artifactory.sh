# #!/bin/bash     
# PROJECT_ID="oro-smart-capex-001-dev"
# REGION="europe-west3"
# REPOSITORY="smart-capex-capacity"


# IMAGE_TAG='smartcapex-pipeline:latest'

# # Configure Docker
# gcloud auth configure-docker $REGION-docker.pkg.dev

# # Check if the repository exists
# check_for_repo=$(gcloud artifacts repositories describe $REPOSITORY --location=$REGION 2>&1 >/dev/null)

# if [[ $check_for_repo == ERROR* ]]; then
#   echo "Creating a repository named $REPOSITORY"
#   # Create repository in the artifact registry
#   gcloud artifacts repositories create $REPOSITORY \
#     --repository-format=docker \
#     --location=$REGION \
#   --description="Vertex AI Training Custom Containers"
#else
# echo "There is already a repository named $REPOSITORY"
#fi

## Push
#docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG


#Docker Push:
docker push europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.1.0
