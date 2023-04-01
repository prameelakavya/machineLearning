#!/bin/bash

IFS='/' #setting space as delimiter  
for filename in $1; do
    read -ra filepathsplit <<<"$filename"
    if [[ $filename[-1]!='manifest' ]]; then
        res="$(wc -l $filename)"
        echo $res
    fi
done
