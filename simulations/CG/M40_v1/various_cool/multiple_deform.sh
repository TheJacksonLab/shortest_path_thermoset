#!/bin/sh

for i in {31..50}
do
sed "s/87287/1994$i/g" in.cool > in.cooltmp
lmp_serial -in in.cooltmp
sed "s/87287/1994$i/g" in.deform > in.deformtmp
lmp_serial -in in.deformtmp
mv dump_relax.data dump_relax$i.data
done


