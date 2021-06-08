#!/bin/bash 

snakemake -s snakes/Etienne.snake --cluster "sbatch --time 48:00:00 -J Etienne --ntasks-per-socket=4 --mem-per-cpu=32G -x w03,w05 --output ./OUT_Etienne/Etienne%j.log --error ./ERR_Etienne/Etienne%j.log" -j 64