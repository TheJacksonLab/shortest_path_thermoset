#!/bin/sh

mkdir various_deform
cp cool.dat various_deform/
for i in {21..50}
do
sed "s/87287/1994$i/g" in.deform > various_deform/in.deformtmp
cd various_deform
lmp_serial -in in.deformtmp
mv dump_relax.data dump_relax$i.data
cd ..
done


