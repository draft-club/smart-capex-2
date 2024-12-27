# Configure GCP VPC Network for Internet Access

This guide describes how to configure a Virtual Private Cloud (VPC) network in Google Cloud Platform (GCP) and enable its instances to access the Internet.

## Prerequisites

- Make sure you have a GCP account with the necessary permissions to create and manage VPC networks, Cloud Routers, Cloud NAT gateways, and firewall rules.

## Steps

### Step 1: Create a VPC network

1. Go to the VPC networks page in the Google Cloud Console.
2. Click "Create VPC network".
3. Enter a "Name" for the network.
4. Under "Subnets", click "Add subnet", enter a "Name", choose a "Region", and specify an "IP address range".
5. Click "Create" to create the VPC network and its subnets.

### Step 2: Create a Cloud Router

1. Go to the Cloud Router page in the Google Cloud Console.
2. Click "Create Router".
3. Choose the same "Region" as the subnet you created.
4. Enter a "Name" for the router.
5. Select the VPC network you created from the "Network" dropdown.
6. Click "Create" to create the router.

### Step 3: Create a Cloud NAT gateway

1. Go to the Cloud NAT page in the Google Cloud Console.
2. Click "Create NAT gateway".
3. Enter a "Name" for the NAT gateway.
4. Choose the same "Region" as the subnet and Cloud Router.
5. Select the Cloud Router you created from the "Router" dropdown.
6. Under "Cloud NAT settings", choose "Manual" for "NAT IP addresses".
7. Click "Reserve a static IP address" to create and associate a new static IP address.
8. Click "Add all eligible subnets to Cloud NAT" to add your subnet to the NAT gateway.
9. Click "Create" to create the NAT gateway.

### Step 4: Create a firewall rule

1. Go to the Firewall page in the Google Cloud Console.
2. Click "Create firewall rule".
3. Enter a "Name" for the rule.
4. Choose the "Network" as the VPC network you created.
5. For "Targets", choose "All instances in the network".
6. For "Source filter", choose "IP ranges".
7. In "Source IP ranges", enter "0.0.0.0/0".
8. Under "Protocols and ports", select "Specified protocols and ports", then enter "tcp:80;tcp:443;icmp" (for HTTP, HTTPS, and ping).
9. Click "Create" to create the rule.

## Conclusion

With this setup, instances in your VPC network can reach the Internet via the Cloud NAT gateway. Outbound connections are possible, and responses to these connections are allowed back in. However, the instances cannot be reached directly from the Internet. If inbound connections are necessary, consider setting up Cloud Load Balancing and/or Cloud VPN. Always adhere to the principle of least privilege when configuring firewall rules.
