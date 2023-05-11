#! /bin/bash

for i in "$@"
do
  cd-hit \
    -i "$i" \
    -o "$i.output"\
    -c 0.4 \
    -n 2 \
    -b 0 \
    -T 0 \
    -d 100
done
