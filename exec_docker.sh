#! /bin/bash

green=`tput setaf 2`
reset=`tput sgr0`

date=$(date '+%Yy-%mm-%dd_%Hh-%Mm-%Ss')
name_image="image-${date}"
name_container="container-${date}"

echo -e "${green}\n\nBuilding docker-image...${reset}"
docker build -t $name_image .

# echo -e "${green}\n\nRemoving additional <none> images...${reset}"
# docker image prune -f

echo -e "${green}\n\nShow all images:${reset}"
docker image ls

# Run Container
echo -e "${green}\n\nRun docker-container:${reset}"
args="$@"
s_path="${PWD}/data"
data_path="/app/data"
docker run \
-it \
--rm \
--gpus all \
--shm-size 8G \
--name $name_container \
--mount type=bind,source=$s_path,target=$data_path \
-i $name_image $args

echo -e "${green}\n\nDelete old image:${reset}"
docker image rm $name_image