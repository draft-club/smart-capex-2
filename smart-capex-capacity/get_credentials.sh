#!/bin/sh -x
#PAYLOAD=$(cat <<EOF
#{
#"audience": "//iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}",
#"grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
#"requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
#"scope": "https://www.googleapis.com/auth/cloud-platform",
#"subjectTokenType": "urn:ietf:params:oauth:token-type:jwt",
#"subjectToken": "${CI_JOB_JWT_V2}"
#}
#"EOF
#)
#"audience": "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}",
PAYLOAD="$(cat <<EOF
{
  
  "audience": "projects/934643879169/locations/global/workloadIdentityPools/orange-pool/providers/gitlab-tech-orange,
  "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
  "requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
  "scope": "https://www.googleapis.com/auth/cloud-platform",
  "subjectTokenType": "urn:ietf:params:oauth:token-type:jwt",
  "subjectToken": "${GITLAB_OIDC_TOKEN}"
}
EOF
)"
echo "${PAYLOAD}"

FEDERATED_TOKEN=$(curl -X POST "https://sts.googleapis.com/v1/token" \
 --header "Accept: application/json" \
 --header "Content-Type: application/json" \
 --data "${PAYLOAD}" \
 | jq -r '.access_token'
 )
ACCESS_TOKEN=$(curl -X POST "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${SERVICE_ACCOUNT_EMAIL}:generateAccessToken" \
--header "Accept: application/json" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer ${FEDERATED_TOKEN}" \
--data '{"scope": ["https://www.googleapis.com/auth/cloud-platform"]}' \
| jq -r '.accessToken'
)
echo "access token"
echo "${ACCESS_TOKEN}"
