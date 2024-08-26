#!/bin/bash
cd /opt/virtuoso-opensource
ARG=( "ld_dir('/import', '*.nt', 'http://localhost:8890/dataspace');" "rdf_loader_run(log_enable => 3);" "checkpoint;" "exit;" )
for a in "${ARG[@]}"; do
  echo "Running $a"
  /opt/virtuoso-opensource/bin/isql exec="$a" ;
done