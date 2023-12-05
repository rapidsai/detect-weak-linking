# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import os.path
import pprint
import re
import shutil
import subprocess
import sys
import tempfile

def execute(command,args, **kwargs):
  working_dir = kwargs.get('cwd', None)
  try:
    invoke = [command]
    invoke += args
    output = subprocess.run(invoke, cwd=working_dir,check=True, capture_output=True).stdout
    output = output.splitlines()
  except (OSError, subprocess.CalledProcessError) as err:
    output = []
  return output

def is_elf(file_path: str) -> bool:
    with open(file_path, 'rb') as f:
      first_byte = f.read(4)
      return first_byte == b'\x7fELF'

def extract_cubins(elf_file: str, dump_location: str):
  execute("cuobjdump", ["--extract-elf", "all", elf_file], cwd=dump_location)
  files = os.listdir(dump_location)
  return [ os.path.join(dump_location, f) for f in files ]

class Symbol:
  def __init__(self, name, raw_type, str_size) -> None:

    self.type = raw_type
    self.name = name
    self.size = int(str_size)

  def __eq__(self, other):
    return self.name == other.name

  def __str__(self):
    return self.name

def extract_info_from_cubin(file: str):
  # Each entry for a symbol has the format "type.<symbol>"
  # we abuse the fact that the symbols will start with an `_`
  # to construct an unique id that allows us to ignore
  # info lines
  regex = re.compile(r"\s+|\._")
  output = execute("size", ["-A",  file])
  nice_symbols = []
  for line in output:
     raw_text = line.decode('utf8')
     entry = regex.split(raw_text)
     if len(entry) == 4:
        nice_symbols.append(Symbol("_"+entry[1], entry[0], entry[2]))
  return nice_symbols


def transform_to_demangled_names(symbols):
  # Call cu++filt with a subset of entries to save time
  # We can't pass all entries as cu++filt has a max
  # input size
  def chunk_iter(list):
    for i in range(0, len(list), 128):
        yield list[i:i + 128]

  all_names = list(symbols.keys())
  all_sizes = list(symbols.values())
  symbols_out = {}
  for names,sizes in zip(chunk_iter(all_names), chunk_iter(all_sizes)):
    names = [n.decode('utf8') for n in execute("c++filt", names)]
    for n,s in zip(names,sizes):
      symbols_out[n]=s

  return symbols_out


class SymbolCache:

  def __init__(self, inclusions) -> None:
    self.has_inclusions = False
    if inclusions:
      # build regex engines
      self.inclusions = [ re.compile(e) for e in inclusions ]
      self.has_inclusions = True
    self.cubin_cache = {}
    self.tmpdir = tempfile.mkdtemp()

  def __del__(self):
    shutil.rmtree(self.tmpdir)

  def load(self, path) -> None:
    if path not in self.cubin_cache and is_elf(path):
      dump_loc = os.path.join(self.tmpdir, os.path.basename(path))
      os.mkdir( dump_loc )
      cubins = extract_cubins(path, dump_loc)
      self.cubin_cache[path] = cubins

  def display_sizes(self):
    def add_or_update(json, symbol):
      if symbol.name in json:
        json[symbol.name] += symbol.size
      else:
        json[symbol.name] = symbol.size

    # determine if a name matches any of the inclusions regex
    def has_match(name, inclusions):
      for regex in inclusions:
        if regex.search(name) :
          return True
      return False

    entries = {}
    for values in self.cubin_cache.values():
      for cubin in values:
        symbols = extract_info_from_cubin(cubin)
        for s in symbols:
          add_or_update(entries, s)

    entries = transform_to_demangled_names(entries)

    # apply the regex filters
    if self.has_inclusions:
      entries = {k:v for k,v in entries.items() if has_match(k, self.inclusions) }

    total_size = 0
    for e in entries.values():
      total_size += e

    for k, v in sorted(entries.items(), key=lambda kv: kv[1]):
      print("%s: %s bytes" % (k,v))
    print("Total uncompressed size of CUDA kernels: ", total_size, "bytes")


def main():
  parser = argparse.ArgumentParser(prog='report CUDA SASS Kernel sizes')
  parser.add_argument("-i", "--include", type=str, nargs='+', help="only include symbols that match this pattern ( applied on demangled names)")
  parser.add_argument("input", nargs='+', type=str, help="elf file ( .so, .exe, .o ) or directory")
  args = parser.parse_args()

  cache = SymbolCache(args.include)

  # Transform any directory into files
  items = []
  for item in args.input:
    if os.path.isdir(item):
      for possible_item in os.listdir(item):
        pitem = os.path.join(item, possible_item)
        if os.path.isfile(pitem):
          items.append(pitem)
    else:
      items.append(item)

  for item in items:
    if os.path.isfile(item):
      cache.load(item)

  if len(cache.cubin_cache) == 0:
    print("Invalid input given")
    parser.print_help()
    sys.exit(1)

  cache.display_sizes()

if __name__ == '__main__':
  main()
