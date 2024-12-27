PROJECT_ID="oro-smart-capex-001-dev"
REGION="europe-west3"
REPOSITORY="smart-capex-capacity/smartcapex-churn"
IMAGE='churn'
IMAGE_TAG='churn:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
