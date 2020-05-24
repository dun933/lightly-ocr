docker rmi -f $(docker images -a | grep "^<none>" | awk '{print $3}')
docker build -t aar0npham/lightly-ocr:latest ocr
