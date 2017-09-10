#!/bin/bash




path="test"


red=$(tput setaf 1)
blue=$(tput setaf 4)
normal=$(tput sgr0)

parse(){
local path=$1
for entry in "$path"/*.dax
do
  echo "Xml file:  ${blue} $entry ${normal}"
  python xmlToDag.py $entry
done
}

parse "instances"
#parse "CYBERSHAKE"
#parse "GENOME"
#parse "LIGO"
#parse "MONTAGE"
#parse "SIPHT"


