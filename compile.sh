#!/bin/bash

(cd build; cmake ../; make; mv HEA ../bin/)

echo "\n\nThe exec file was written in the folder bin/"