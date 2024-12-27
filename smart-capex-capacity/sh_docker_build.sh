#!/bin/bash     
PROJECT_ID="oro-smart-capex-001-dev"
REGION="europe-west3"
REPOSITORY="smart-capex-capacity"
IMAGE='smartcapex-pipeline'
IMAGE_TAG='smartcapex-pipeline:latest'

#docker build -t $IMAGE .
#docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG





#Docker Build:
docker build . -t smart-capex-capacity-pipeline #--build-arg http_proxy=http://kirk.crm.orange.intra:3128 --build-arg https_proxy=http://kirk.crm.orange.intra:3128 --build-arg NO_PROXY=metadata.google.internal,bigquery.googleapis.com,*.googleapis.com

#Docker Tag:
docker tag smart-capex-capacity-pipeline europe-west3-docker.pkg.dev/oro-smart-capex-001-dev/smart-capex-capacity/smartcapex-pipeline/smart-capex-capacity-pipeline-image:0.1.0
 
