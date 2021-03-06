#!/bin/sh
# Assumes python3-runner is created and named 'python3-runner'

docker run --rm -v $PWD:/opt/sources python3-runner /bin/sh -c \
	'make -C set-steering clean && make -C set-steering && \
	 make -C fast clean && make -C fast && \
	 make -C intersection clean && make -C intersection'
