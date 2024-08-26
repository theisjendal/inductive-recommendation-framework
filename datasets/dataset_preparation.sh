#!/bin/bash

echo "NOTICE PLEASE READ IMPORTANT: GET PERMISSION TO USE DATASETS, SEE README.MD"

RUN_DOWNLOADERS=true
RUN_CONVERTERS=true
RUN_PREPROCESSORS=true
RUN_PARTITIONERS=true
RUN_GRAPH_BUILDERS=true
RUN_FEATURE_EXTRACTORS=true

mkdir -p mindreader movielens ml-mr ml-mr-1m
mkdir -p amazon-book

set -e
if $RUN_DOWNLOADERS; then
  ### Convert YK as needs conversion first
  cd converters
  python3 ykg-converter.py


  ########## Download ##########
  cd ../downloaders

  # Yelp
  python3 yelpkg_downloader.py

  # Mindreader
  python3 mr-downloader.py

  # Movielens (Repeat for safety as internet connection may be unstable)
  python3 ml-downloader.py
  python3 ml-downloader.py
  python3 ml-downloader.py

  # Join ml and mr
  python3 ml-mr-downloader.py

  # Amazon-Book
  python3 ab-downloader-kgat.py
  python3 ab-downloader-og.py
else
  cd downloaders
fi

########## Convert ##########
if $RUN_CONVERTERS; then
  # Movielens
  cd ../converters
  python3 mr-converter.py --path ../ml-mr

  cd ../downloaders
  python3 mr-kg-text.py --dataset ml-mr

  # copy ml-mr to ml-mr-1m
  rsync -a --relative --info=progress2 ../ml-mr/./* ../ml-mr-1m
  cd ../converters

  # Amazon-Book
  python3 ab_converter_og.py
  python3 ab_converter_kgat.py
else
  cd ../converters
fi


if $RUN_PREPROCESSORS; then
  ########## Preprocess ##########
  # Yelp
  cd ../preprocessors
  python3 preprocessor.py --path .. --dataset yelp

  # Movielens
  python3 preprocessor.py --path .. --dataset ml-mr
  python3 preprocessor.py --path .. --dataset ml-mr-1m

  # Amazon-Book
  python3 preprocessor.py --path .. --dataset amazon-book
else
  cd ../preprocessors
fi

if $RUN_PARTITIONERS; then
  ########## Partition ##########
  cd ../partitioners

  # Yelp
  python3 partitioner.py --path .. --experiment yk_temporal

  # Movielens
  python3 partitioner.py --path .. --experiment ml_temporal
  python3 partitioner.py --path .. --experiment ml_1m_temporal

  # Amazon-Book
  python3 partitioner.py --path .. --experiment ab_temporal
else
  cd ../partitioners
fi

if $RUN_GRAPH_BUILDERS; then
  ########## Create DGL graphs ##########
  cd ../converters

  # Yelp
  python3 experiment_to_dgl.py --path .. --experiment yk_temporal

  # Movielens
  python3 experiment_to_dgl.py --path .. --experiment ml_temporal
  python3 experiment_to_dgl.py --path .. --experiment ml_1m_temporal

  # Amazon-Book
  python3 experiment_to_dgl.py --path .. --experiment ab_temporal
else
  cd ../converters
fi

if $RUN_FEATURE_EXTRACTORS; then
  ########## Create features ##########
  cd ../feature_extractors
  sh fscript.sh
else
  cd ../feature_extractors
fi

