noneContainter=$(docker images -a | grep "^<none>" | awk '{print $3}')
if [ ! -z "$noneContainter" ]; then
    docker rmi -f $noneContainter
fi
docker build -t aar0npham/lightly-ocr:latest ocr
docker run -p 5000:5000 aar0npham/lightly-ocr:latest 
