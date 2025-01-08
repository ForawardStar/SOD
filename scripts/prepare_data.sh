set -o xtrace

# download data from github address
wget -c --no-check-certificate --content-disposition https://github.com/yuhuan-wu/EDN/releases/download/v1.0/SOD_datasets.zip -O SOD_datasets.zip
unzip -n SOD_datasets.zip -d data/
