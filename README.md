# fx-cmix
The fx-cmix is a updated implementation of [fast-cmix](https://github.com/saurabhk/fast-cmix/). 

Prize awarded on February 2, 2024. http://prize.hutter1.net/

# Submission Description
This submission contains fallowing modifications on top of the recent fast-cmix Hutter Prize winner:
* paq8hp model is replaced with fxcmv1 model with fallowing notable additions:
  * Multiple state tables are used in predictors, this allows better predictability.
  * Most contexts are divided between 30 main predictors, this allows more efficient memory usage per context.
  * Added bracket, quote, first char, char in paragraph, column, table, template, word stream/paragraph context. These contexts are parsed depending on input, this includes parsing of wiki links, http links, tables, columns, paragraphs, quotes, brackets, list, templates.
  * Some contexts are swapped in predictors depending on what current input is (table, column mode, word/paragraph, list).
  * Some predictors are switched on/off depending on last char, link or current bracket which improves compression.
  * Predictions are mixed with context that are more aware of what predictors are outputting.
  * Match model (not present in paq8hp model).
  * Predictors are faster allowing more complex contexts.
* new dictionary
* small change in phda9 preprocessor and in two tables in cmix
* memory usage is larger in fxcmv1 model compared to old paq8hp

Below is the fx-cmix result:

| Metric | Value |
| --- | ----------- |
| fx-cmix compressor's executable file size (S1)| 436707 bytes |
| fx-cmix self-extracting archive size (S2)| 112148343 bytes |
| Total size (S) | 112585050 bytes |
| Previous record (L) | 114156155 bytes |
| fx-cmix improvement (1 - S/L) | 1.38% |

| Experiment platform |  |
| --- | ----------- |
| Operating system | Ubuntu 20.04 |
| Processor | Intel(R) Xeon(R) CPU @ 2.20GHz [Geekbench score 706](https://browser.geekbench.com/v5/cpu/21976774/claim?key=736235)|
| Memory | 32 GB |
| Decompression running time | 85 hours |
| Decompression RAM max usage | 9575124 KiB |
| Decompression disk usage | ~35GB |

Time, disk, and RAM usage are approximately symmetric for compression and decompression.


# Instructions
The installation and usage instructions for fx-cmix are the same as for fast-cmix.

One important note: it is recommended to change one variable in the source code for PPM. From line 26 in src/models/ppmd.cpp:

```
// If mmap_to_disk is set to false (recommended setting), PPM will only use RAM
// for memory.
// If mmap_to_disk is set to true, PPM memory will be saved to disk using mmap.
// This will reduce RAM usage, but will be slower as well. *Warning*: this will
// write a *lot* of data to disk, so can reduce the lifespan of SSDs. Not
// recommended for normal usage.
bool mmap_to_disk = true;
```

This variable is set to true by default, to comply with the Hutter Prize RAM limit.

# Installing packages required for compiling fx-cmix compressor from sources on Ubuntu
Building fx-cmix compressor from sources requires clang-17, upx-ucl, and make packages.
On Ubuntu, these packages can be installed by running the following scripts:
```bash
./install_tools/install_upx.sh
./install_tools/install_clang-17.sh
```

# Compiling fx-cmix compressor from sources
A bash script is provided for compiling cmix-hp compressor from sources on Ubuntu. This script places the cmix-hp executable file named as `cmix` in `./run` directory. The script can be run as
```bash
./build_and_construct_comp.sh
```

# Running fx-cmix compressor
To run the cmix-hp compressor use
```bash
cd ./run
cmix -e <PATH_TO_ENWIK9> enwik9.comp
```


# Running fx-cmix decompressor
The compressor is expected to output an executable file named `archive9` in the same directory (`./run`). The file `archive9` when executed is expected to reproduce the original enwik9 as a file named `enwik9_restored`. The executable file `archive9` should be launched without argments from the directory containing it. 
```bash
cd ./run
./archive9
```
