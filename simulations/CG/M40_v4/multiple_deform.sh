#!/bin/sh

mkdir hT_deform
cp cool.dat hT_deform/
for i in {1..20}
do
cd hT_deform
sed "s/87287/1994$i/g" ../in.deform > in.deformtmp
sed -i '' "s/0\.1/0.4/g" in.deformtmp
lmp_serial -in in.deformtmp
mv dump_relax.data dump_relax$i.data
cd ..
done


