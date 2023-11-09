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
import json
import os
import os.path
import re
import subprocess
import sys

def execute(command, args):
  try:
    invoke = [command]
    invoke += args
    output = subprocess.run(invoke, check=True, capture_output=True).stdout
    output = output.splitlines()
  except (OSError, subprocess.CalledProcessError) as err:
    output = []
  return output

def get_external_symbols(elf_file: str) -> str:
  # Need to figure out a way to tell if we need to add `-D`
  # try again if output has `no symbols`?
  nm_output = execute("nm", ["-g", "--defined-only", elf_file])
  if len(nm_output) == 0:
    nm_output = execute("nm", ["-D", "-g", "--defined-only", elf_file])

  # split on new line
  nice_symbols = [Symbol.from_nm(x) for x in nm_output]
  return nice_symbols

def get_cuda_entry_symbols(elf_file: str, no_ptx: bool):

  options = ["-symbols", "-ptx", elf_file]
  if no_ptx:
    options = ["-symbols", elf_file]

  cuobjdump_output = execute("cuobjdump", options)

  # lines that start with `STO_ENTRY` + STB_GLOBAL are possible public sass entry points
  # lines that start with `STT_FUNC` + STB_LOCAL are private sass entry points
  # lines that start with `.visible .entry` are possible public ptx
  # lines that start with `.entry` are private ptx
  #
  # For public entry points we have to check the symbol visibility on the host side
  # function as well
  private_entry_points = []
  public_entry_points  = []

  for raw_line in cuobjdump_output:
    line = raw_line.decode('utf8')

    split_line =line.split(" ")
    if "STO_ENTRY" in line:
      if "STB_GLOBAL" in line:
        public_entry_points.append(Symbol("SASS", split_line[-1]))
      else:
        private_entry_points.append(Symbol("sass", split_line[-1]))

    if ".entry" in line:
      # For ptx entries the name line has a trailing '('
      # that we need to drop
      name = split_line[-1][:-1]
      if ".visible" in line:
        public_entry_points.append(Symbol("PTX", name))
      else:
        private_entry_points.append(Symbol("ptx", name))

  return public_entry_points, private_entry_points

class Symbol:
  def __init__(self, mode : str, name : str) -> None:
    self.mode = mode
    self.name = name

  @classmethod
  def from_nm(cls, raw_text : str):
    # split on space
    # b'000000000000f234 W _ZN3cub19DeviceCountUncachedEv'
    address, mode, name = raw_text.decode('utf8').split(" ")
    return cls(mode, name)

  def __eq__(self, other):
    return self.name == other.name

  def __str__(self):
    return self.name

class ElfEntity:
  r"""
  A DSO or executable with a record of all external symbols
  """
  def __init__(self, file_path: str, no_ptx: bool) -> None:
    print("Extracting symbols from", file_path)
    self.file = file_path
    self.all_symbols = get_external_symbols(self.file)
    self.cuda_public_entry_symbols, self.cuda_private_entry_symbols = get_cuda_entry_symbols(self.file, no_ptx)

  @staticmethod
  def is_elf(file_path: str) -> bool:
    with open(file_path, 'rb') as f:
      first_byte = f.read(4)
      return first_byte == b'\x7fELF'

class ElfCache:
  """
  Keep ELF information in a cache so we
  don't need to keep requerying when
  doing cross library checks
  """

  def __init__(self, multiple_entries: bool, show_variables: bool, exclusions, no_ptx: bool) -> None:
    self.show_only_multiple_entries = multiple_entries
    self.include_u_variables = show_variables
    self.has_exclusions = False
    self.no_ptx = no_ptx

    if exclusions:
      # build regex engines
      self.exclusions = [ re.compile(e) for e in exclusions ]
      self.has_exclusions = True

    self.cache = {}

  def load(self, path, with_ldd) -> None:
    if path not in self.cache:
      if ElfEntity.is_elf(path):
        self.cache[path] = ElfEntity(path, self.no_ptx)

        if with_ldd:
          self.load_deps(path)

  def load_deps(self, path):
    # run objdump -p
    # extract RPATH and RUNPATH entries
    # extract NEEDED entries
    objdump_output = execute("objdump", ["-p", path])

    libnames = []
    directories = []
    for line in objdump_output:
      line = line.decode('utf8')
      if 'NEEDED' in line:
        libnames.append(line.split(" ")[-1])
      if 'RUNPATH' in line or 'RPATH' in line:
        entries = line.split(" ")[-1]
        if entries == "$ORIGIN":
          directories.append(os.path.dirname(path))
        else:
          entries = entries.split(":") #Can have multiple values in the same entry
          directories+=entries

    for lib in libnames:
      lib_to_load = [ os.path.join(dir,lib) for dir in directories if os.path.exists(os.path.join(dir,lib)) ]
      if len(lib_to_load) > 0:
        # use the first one since that should be the first in RPATH/RUNPATH order
        self.load(lib_to_load[0], True)

  def find_issues(self, all_weak: bool):
    # Report all bad SASS/PTX entries
    #
    # Should display c++filt names
    # use https://docs.python.org/3/library/ctypes.html
    # libc.so.6
    #
    def add_or_update(json, symbol, file):
      # Layout should be json
      # { "symbol" : {
      #     "demangled_name" : "niiice"  # added as a post filter
      #     "<file_path>" : ["sass", "sass"]
      #   },
      # }
      # Will need to c++filt the names I expect
      if symbol.name in json:
        entry = json[symbol.name]
        if file in entry:
          entry[file].append(symbol.mode)
        else:
          entry[file]=[symbol.mode]
      else:
        json[symbol.name] = { file: [symbol.mode] }

    # determine if a name matches any of the exclusion regex
    def has_match(name, exclusions):
      for regex in exclusions:
        if regex.search(name) :
          return True
      return False

    # Add a demangled name entry into each entry
    def add_demangled_names(json_e):
      # Call cu++filt with every entry in a single invocation to save time
      names = [ n for n in json_e ]
      cufilt_output = names
      if len(names) > 0:
        cufilt_output = execute("c++filt", names)
        cufilt_output = [n.decode('utf8') for n in cufilt_output]

      for k,n in zip(json_e,cufilt_output):
        json_e[k]['symbol']=n

      return json_e

    json_entries = {}
    for key in self.cache:
      entity = self.cache[key]

      if all_weak:
        for entry in entity.all_symbols:
          add_or_update(json_entries, entry, key)
      else:
        for entry in entity.cuda_public_entry_symbols:
          if entry in entity.all_symbols:
            add_or_update(json_entries, entry, key)

      if self.include_u_variables:
        for entry in entity.all_symbols:
          if entry.mode == "u":
            add_or_update(json_entries, entry, key)

    # apply multiple_entries filter
    if self.show_only_multiple_entries:
      json_entries = {k:v for k,v in json_entries.items() if len(v) > 1}

    # Add the demangled names, needs to be before regex filtering
    # as that is done on the symbol value
    add_demangled_names(json_entries)

    # apply the regex filters
    if self.has_exclusions:
      json_entries = {k:v for k,v in json_entries.items() if not has_match(v['symbol'], self.exclusions) }

    return json_entries

def remove_baseline_entries(issues, baseline):
  # This isn't a straight intersection where we drop any
  # symbol with the same name between the two files.
  #
  # What we want to do is the following:
  # Any symbol in `issues` not in `baseline` we propagate.
  # Any symbol in both we keep if the files names differ,
  # or if the count per file differ
  #
  def normalize_names(entry):
    normalized = { os.path.basename(k):v for k,v in entry.items() if not k == "symbol" }
    return normalized

  def intersect_symbol(issue_s, base_s):
    result = {}
    for k in issue_s:
      if not k in base_s:
        result[k]=issue_s[k]
      elif len(issue_s[k]) > len(base_s[k]):
        result[k]=issue_s[k]
    return result

  #
  subset = {}
  for symbol in issues:
    if symbol not in baseline:
      subset[symbol] = issues[symbol]
    else:
      normalized_issue = normalize_names(issues[symbol])
      normalized_base = normalize_names(baseline[symbol])
      possible = intersect_symbol(normalized_issue, normalized_base)
      if len(possible) > 0:
        subset[symbol] = possible

  return subset

# project -r <exe|lib> # follow ldd
# project <exe|lib> # just the item
# project <libA> <libB> # just the listed items
# project <directory> # all libraries in the directory
# project -r <libA> <libB> # recursive across multiple items
# project -r <libA> -b <json_entry> # Only show entries not in the baseline file
def main():
  parser = argparse.ArgumentParser(prog='detect cuda __global__ weak symbols')
  parser.add_argument("-r", dest="recursive", action='store_true', help="also load ldd dependencies")
  parser.add_argument("-u", dest="global_vars", action='store_true', help="show global unique variables")
  parser.add_argument("-m", dest="multiple_entries", action='store_true', help="only show symbols that are in multiple files")
  parser.add_argument("--no-ptx", dest="no_ptx", action='store_true', help="Don't looks for PTX kernel entries")
  parser.add_argument("--all-weak", dest="all_weak", action='store_true', help="Consider all weak/global symbols not just CUDA kernels")
  parser.add_argument("-e", "--exclude", type=str, nargs='+', help="exclude symbols that match this pattern ( applied on demangled names)")
  parser.add_argument("-b", "--baseline", type=argparse.FileType('r'), help="show only results that are not in the baseline file")
  parser.add_argument("input", nargs='+', type=str, help="elf file ( .so, .exe, .o ) or directory")
  args = parser.parse_args()

  cache = ElfCache(args.multiple_entries, args.global_vars, args.exclude, args.no_ptx)

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
      cache.load(item, args.recursive)

  if len(cache.cache) == 0:
    print("Invalid input given")
    parser.print_help()
    sys.exit(1)

  issues = cache.find_issues(args.all_weak)

  # Load up the baseline json
  if args.baseline:
    baseline = json.loads(args.baseline.read())
    issues = remove_baseline_entries(issues, baseline)

  if issues:
    if len(issues) >= 1:
      print(json.dumps(issues, indent=2))
      sys.exit(1)
  return 0

if __name__ == '__main__':
  main()
