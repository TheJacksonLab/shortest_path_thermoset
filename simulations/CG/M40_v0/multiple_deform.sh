#!/bin/sh

for i in {1..3}
do
sed "s/87287/1994$i/g" in.deform > in.deformtmp
lmp_serial -in in.deformtmp
mv dump_relax.data dump_relax$i.data
done


