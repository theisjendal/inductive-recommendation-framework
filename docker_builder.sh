#!/bin/bash

set -e


if [ ! -d "datasets/yelpkg/The-Yelp-Collaborative-Knowledge-Graph" ]; then
  mkdir -p datasets/yelpkg/
  cd datasets/yelpkg
  mkdir database data
  git clone https://github.com/MadsCorfixen/The-Yelp-Collaborative-Knowledge-Graph.git
else
  cd datasets/yelpkg
fi

### Virtuoso setup

if [ ! -f "data/yelp_dataset.tar" ]; then
    echo "You must download the yelp dataset from: https://www.yelp.com/dataset"
    echo "Place it in the folder: yelpkg/repos/The-Yelp-Collaborative-Knowledge-Graph/data"
    exit 1
fi

if [ ! -f "data/yelp_academic_dataset_user.json" ]; then
  cd data
  tar -xv --skip-old-files -f yelp_dataset.tar
  cd ..
fi

cd The-Yelp-Collaborative-Knowledge-Graph

if [ ! -d ".venv" ]; then
  python3.10 -m venv .venv
  source .venv/bin/activate
  pip install setuptools
  pip install -r requirements.txt
  deactivate
fi
if [ ! -f "../data/yelp_business.nt" ]; then
  echo "Creating YCKG"
  source .venv/bin/activate
  cp UtilityData/* ../data/
  python3 create_YCKG.py --read_dir '../data' --write_dir '../data' --include_schema True --include_wikidata True &&
  deactivate
fi

# Go to dataset folder
cd ../..
chmod +x virtuoso_starter_script.sh

# Go to project folder
cd ..

echo $(pwd)


#docker build -f Dockerfile.base -t cuda11 .

docker compose up -d

docker exec virtuoso /app/start_script.sh
docker exec -it trainer /bin/bash
