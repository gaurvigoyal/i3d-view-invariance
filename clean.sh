#!/bin/bash

echo "Are you want to clean this directory?(y/n)"
read varname
case $varname in
    [Yy]* ) 	rm -R ./../results/checkpoints/$1/f*
				rm -R ../results/graphs/$1/*
				rm -R ../results/out/$1/*;;
    [Nn]* ) echo Cancelled; exit;;
    * ) echo "Response invalid. Cancelled.";;
esac
