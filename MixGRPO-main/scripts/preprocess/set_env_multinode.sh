#!/bin/bash

awk '{print $0}' data/hosts/hostfile | nl -v 0 | while read index node; do
    pdsh -w $node "echo export INDEX_CUSTOM=$index >> ~/.bashrc"
done
