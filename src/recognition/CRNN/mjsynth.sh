#!/bin/bash

# https://stackoverflow.com/a/21189044
function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

eval $(parse_yaml config.yml)

eval $(cd /mnt/Vault/database/)
if [ ! -f "mjsynth.tar.gz" ]; then
    wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
    tar xvzf mjsynth.tar.gz
    mv mnt/ramdisk/max/90kDICT32px $DATA_PATH
    rm -r mnt
fi
eval $(cd ~/Documents/cs/lightly-ocr/src/recognition/CRNN)

echo "Done processing mjsynth"





