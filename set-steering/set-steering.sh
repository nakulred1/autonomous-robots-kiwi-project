#!/bin/sh

docker run --rm -ti --net=host -v $PWD:/opt/sources python3-runner \
	./set-steering.py $@
