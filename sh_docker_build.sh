PROJECT_ID="oro-smart-capex-001-dev"
REGION="europe-west3"
REPOSITORY="smart-capex-capacity/smartcapex-dataquality"
IMAGE='ocdvt'
IMAGE_TAG='ocdvt:dqlatest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG
