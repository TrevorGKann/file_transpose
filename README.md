A quick demo I made to see how fast different methods are of transposing a large matrix and writing it to disk. 
When tranposing a very large matrix (too big to fit in RAM), you need sparse accesses to the durable file, meaning writes are very slow and un-optimized from a sys-call perspective. 
Pretty much any method (particularly naive/canonical ones) is much slower than you'd expect. 

## Use

There's a simple CLI made from clap.
The program will automatically generate a file to tranpose with size equal to 2 to the power of the CLI argument; e.g., `matrix_transposer 30` will make a 2^30=1 GiB file to tranpose. 
If you want to keep the resulting files around you can use the `-k` flag, but I'd caution against this when your files get huge unless you want to chew up all your memory. 
You'll need at least 2x the size of the input file available on your disk to run this. 

Make sure to build the project in release mode `cargo build --profile=profiling` so performance isn't biased by slow code. 

Flags can be provided to test different methods, not all will work on all file sizes:

| flag     | description                                                                                                                                                                                                    |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -i       | do the transpose in memory (aka easy solution that doesn't scale). You will need RAM greater than the size of the input file for this to work                                                                  |
| -m       | memmap'ed solution. Opens the file as a mmap and does the tranpose there. This should bypass some syscalls of doing it straight to disk                                                                        |
| -j       | read all the rows of the matrix and splice them up into temporary column files, then concatenate all the temp files together                                                                                   |
| -o       | naively do the entire transpose on disk                                                                                                                                                                        | 
| -b       | do the entire transpose on disk but buffer in several columns to write at once. This removes a lot of the syscall overhead by buffering                                                                        |
| -a       | **toggle** all solutions on. Doing `-a` then another solution will run all but that solution.                                                                                                                  |
| -k       | keep around resulting files after use. Warning: this takes up a lot of disk space for large files                                                                                                              |                                             
| -c       | Perform runtime assertions that the transposes are correct. This is more for debugging than speed testing. <br/>The in-memory solution is the reference correct one so that one has to be run for this to work |
| -t <num> | Run the tests `num` times                                                                                                                                                                                      |
| -v       | show little samples of the files after running                                                                                                                                                                 |

### Example uses

Do all transpose methods 4 times on a 1 Kib file
``` 
./target/profiling/matrix_transposer -a -t 4 10
```


Run the mmap and buffered solution on a 1 GiB file
``` 
./target/profiling/matrix_transposer -m -b 30
```


Check if the on-disk solution is correct on a 2^15 byte file
``` 
./target/profiling/matrix_transposer -c -i -o 15
```
