#!/usr/bin/fish
echo "zero to empty file"
time dd if=/dev/zero of=empty-file bs=1MB count=1024

echo "random to empty file"
time dd if=/dev/urandom of=random-data bs=1MB count=1024

echo "syncing and flushing caches"
sync
echo 3 > sudo /proc/sys/vm/drop_caches

echo "reading zeros from disk"
time dd if=empty-file of=empty-file2 bs=1MB

echo "reading random from disk"
time dd if=random-data of=random-data2 bs=1MB
