#!/bin/bash
# Tests of the PKC reader (io/pkc_core.c). usage: ./run_tests.sh <scratch dir> [pkc_to_gadget binary]
# Writes version-3 test snapshots with the reference library, then checks the C reader against it.
set -u
OUT=${1:?usage: run_tests.sh <scratch dir> [pkc_to_gadget]}
P2G=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
MK=$HERE/make_testset
CMP=$HERE/pkc_compare
mkdir -p "$OUT"
fails=0
check() { # name, expected exit code, command...
  local name=$1 want=$2; shift 2
  local out; out=$("$@" 2>&1); local rc=$?
  if [ "$rc" = "$want" ]; then echo "PASS $name"; else echo "FAIL $name (exit $rc, want $want)"; fails=$((fails+1)); fi
  echo "$out" | sed 's/^/     /'
}

# sets: name N ids writers pot block
SETS="A 32 lattice 2 1 1000
B 24 raw 1 0 4096
C 24 delta 3 0 500
D 48 lattice 1 0 65536"
while read -r name N ids W pot B; do
  rm -f "$OUT/$name".*
  check "make $name" 0 "$MK" "$OUT/$name" "$N" "$ids" "$W" "$pot" "$B"
  check "decode $name" 0 "$CMP" decode "$OUT/$name"
  check "shares $name" 0 "$CMP" shares "$OUT/$name" 1 7 64 1000
done <<< "$SETS"

check "subsel A 1" 0 "$CMP" subsel "$OUT/A" 1
check "subsel A 1,3,5,9,10" 0 "$CMP" subsel "$OUT/A" 1,3,5,9,10
check "subsel A 2 (no such subsample)" 1 "$CMP" subsel "$OUT/A" 2
# w0-wa run: Omega_lambda 0, Omega_DE 0.69 -> Ol 0.69, W0 -0.9, WA 0.1; E_z at z = 0.5 from make_testset
EZ=$("$MK" "$OUT/tmpE" 4 lattice 1 0 64 | sed 's/.*E_z //'); rm -f "$OUT"/tmpE.*
check "cosmo A" 0 "$CMP" cosmo "$OUT/A" 0.69 -0.9 0.1 "$EZ"

# corruption and incomplete sets (copies of A)
corrupt() { rm -rf "$OUT/bad"; mkdir -p "$OUT/bad"; cp "$OUT"/A.*.pkc "$OUT/bad/"; }
F=$OUT/bad/A.s25.0.pkc
corrupt; python3 -c "import os,sys; f=sys.argv[1]; os.truncate(f, os.path.getsize(f)-32)" "$F"
check "reject: missing trailer" 1 "$CMP" open "$OUT/bad/A"
corrupt; rm "$F"
check "reject: missing writer file" 1 "$CMP" open "$OUT/bad/A"
flip() { # file, stream number: flip one byte in the middle of that stream's zstd frame (first block)
python3 - "$1" "$2" <<'EOF'
import struct, sys
f, k = sys.argv[1], int(sys.argv[2])
b = bytearray(open(f, 'rb').read())
idx = struct.unpack_from('<Q', b, len(b) - 24)[0]
off = struct.unpack_from('<Q', b, idx)[0]             # first block
ns = b[off + 12]
data = off + 16 + 12 * ns
for j in range(k):
    data += struct.unpack_from('<I', b, off + 16 + 12 * j + 8)[0]
cs = struct.unpack_from('<I', b, off + 16 + 12 * k + 8)[0]
b[data + cs // 2] ^= 0xFF
open(f, 'wb').write(b)
EOF
}
corrupt; flip "$F" 5
check "reject: flipped byte in the velocity stream" 1 "$CMP" open "$OUT/bad/A"
# the potential stream (last one in set A) is not decoded by the reader (Rockstar does not use it)
corrupt; flip "$F" 11
check "ignored: flipped byte in the potential stream" 0 "$CMP" open "$OUT/bad/A"
rm -rf "$OUT/bad"; mkdir -p "$OUT/bad"; cp "$OUT"/D.*.pkc "$OUT/bad/"; flip "$OUT/bad/D.s25.0.pkc" 8
check "reject: flipped byte in the ID stream" 1 "$CMP" open "$OUT/bad/D"
corrupt; python3 - "$F" <<'EOF'
import struct, sys
f = sys.argv[1]; b = bytearray(open(f, 'rb').read())
struct.pack_into('<d', b, 72, 51.0)                   # box_size of one file
open(f, 'wb').write(b)
EOF
check "reject: header from another snapshot" 1 "$CMP" open "$OUT/bad/A"
corrupt; python3 - "$F" <<'EOF'
import struct, sys
f = sys.argv[1]; b = bytearray(open(f, 'rb').read())
struct.pack_into('<I', b, 8, 2)                       # format version 2
open(f, 'wb').write(b)
EOF
check "reject: format version 2" 1 "$CMP" open "$OUT/bad/A"
corrupt; cp "$OUT/bad/A.s25.0.pkc" "$OUT/bad/A.s25.0x.pkc"
check "reject: duplicate (subsample, writer)" 1 "$CMP" open "$OUT/bad/A"
rm -rf "$OUT/bad"

# floats bit-identical to Rockstar reading pkc_to_gadget output (if the converter is given)
if [ -n "$P2G" ]; then
  for name in A D; do
    rm -f "$OUT/g$name".*
    check "pkc_to_gadget $name" 0 "$P2G" --threads 4 --max-per-file 20000 "$OUT/$name" "$OUT/g$name"
    check "gadget floats $name" 0 "$CMP" gadget "$OUT/$name" "$OUT/g$name"
  done
fi

echo "== $fails failure(s)"
exit $((fails > 0))
