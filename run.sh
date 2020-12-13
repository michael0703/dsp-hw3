#!/bin/bash
dir_path="input_file/"
make
./mydisambig $dir_path"input.txt" $dir_path"ZhuYin-Big5.map" $dir_path"bigram-lm.txt" "out.txt"
