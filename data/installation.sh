# curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-471.0.0-linux-x86_64.tar.gz
# tar -xf google-cloud-cli-471.0.0-linux-x86_64.tar
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init
pip install --upgrade google-auth