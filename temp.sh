#!/bin/bash
remote_origin=$(git config --get remote.origin.url)
b=${remote_origin/*svn/https://$(whoami)@svn}
echo $b

