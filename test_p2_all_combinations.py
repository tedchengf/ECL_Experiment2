import numpy as np
import time
import random
from itertools import product 
from psychopy import visual, core, data, event, sound
import gc
import os

import stimuli

# Exact same setup as EXP_4_Perm.py
R = 2
GENERATION_MODE = "Multiset Permutation"

# Formula Pool - exact same as EXP_4_Perm.py
P2 = [
    [
      ("=1", "=0"),
      ("+0", "+0"),
      ("+0", "+0"),
    ]
    ,
    [
      ("+0", "+0"),
      ("=1", "=0"),
      ("+0", "+0"),
    ]
 ]

# Testing Formula - EXACT same as EXP_4_Perm.py lines 100-111
test_SIG = stimuli.Sigma([["r", "b"], ["c","t"], ["l","s"]], ["fill", "shape", "size"], R, generation_mode = "Multiset Permutation")

print("=" * 100)
print("TESTING P2 RULE - ALL COMBINATIONS")
print("=" * 100)
print()
print(f"Total sequences: {len(test_SIG.sequences)}")
print()

test_conj = test_SIG.form_conjunct(P2, conjunct_type="Seq")

# Test all sequences
accepted_seqs = []
rejected_seqs = []

print("Testing all sequences:")
print("=" * 100)

for seq in test_SIG.sequences:
    accepts = test_conj.accepts(seq)
    summary = seq.summarize()  # Using same method as EXP_4_Perm.py
    
    # Get object features for verification
    obj0_summary = summary[0]  # First object
    obj1_summary = summary[1]  # Second object
    
    # Check if first object is red
    obj0_is_red = obj0_summary.get(('fill', 'r'), 0) == 1
    obj0_is_blue = obj0_summary.get(('fill', 'b'), 0) == 1
    
    # Check if second object is circle
    obj1_is_circle = obj1_summary.get(('shape', 'c'), 0) == 1
    obj1_is_triangle = obj1_summary.get(('shape', 't'), 0) == 1
    
    # Expected rule: first=red AND second=circle
    expected_accept = obj0_is_red and obj1_is_circle
    
    result = {
        'seq': seq,
        'pid': seq.pid,
        'accepts': accepts,
        'summary': summary,
        'obj0_summary': obj0_summary,
        'obj1_summary': obj1_summary,
        'obj0_is_red': obj0_is_red,
        'obj1_is_circle': obj1_is_circle,
        'expected_accept': expected_accept,
        'matches': accepts == expected_accept
    }
    
    if accepts:
        accepted_seqs.append(result)
    else:
        rejected_seqs.append(result)
    
    # Print like EXP_4_Perm.py does
    print(f"{accepts} {summary}")

print()
print("=" * 100)
print(f"Total accepted: {len(accepted_seqs)}")
print(f"Total rejected: {len(rejected_seqs)}")
print()

# Verify all sequences
print("=" * 100)
print("VERIFICATION: Checking if rule matches 'First=Red AND Second=Circle'")
print("=" * 100)

mismatches = []
for result in accepted_seqs:
    if not result['expected_accept']:
        mismatches.append(('accepted but should reject', result))
for result in rejected_seqs:
    if result['expected_accept']:
        mismatches.append(('rejected but should accept', result))

if mismatches:
    print(f"\n⚠️  FOUND {len(mismatches)} MISMATCHES:")
    for error_type, result in mismatches:
        print(f"\n{error_type}:")
        print(f"  PID: {result['pid']}")
        print(f"  Formula accepts: {result['accepts']}")
        print(f"  Expected accepts: {result['expected_accept']}")
        print(f"  First object: {result['obj0_summary']}")
        print(f"  Second object: {result['obj1_summary']}")
        print(f"  First is red: {result['obj0_is_red']}")
        print(f"  Second is circle: {result['obj1_is_circle']}")
else:
    print("\n✓ ALL SEQUENCES MATCH EXPECTED BEHAVIOR!")
    print("  Rule is working correctly: First=Red AND Second=Circle")

print()
print("=" * 100)
print("DETAILED BREAKDOWN")
print("=" * 100)

# Show all accepted with hierarchical representation
print(f"\nACCEPTED SEQUENCES ({len(accepted_seqs)}):")
print("-" * 100)
for i, result in enumerate(accepted_seqs, 1):
    print(f"\n{i}. Sequence PID: {result['pid']}")
    print(f"   Hierarchical representation:")
    print(result['seq'].hierarchical_rep(level="Object"))
    print(f"   Summary: {result['summary']}")
    print(f"   ✓ First is red: {result['obj0_is_red']}, Second is circle: {result['obj1_is_circle']}")

# Show all rejected (first 20)
print(f"\nREJECTED SEQUENCES (showing first 20 of {len(rejected_seqs)}):")
print("-" * 100)
for i, result in enumerate(rejected_seqs[:20], 1):
    print(f"\n{i}. Sequence PID: {result['pid']}")
    print(f"   Summary: {result['summary']}")
    print(f"   First is red: {result['obj0_is_red']}, Second is circle: {result['obj1_is_circle']}")
    if not result['obj0_is_red']:
        print(f"   ✗ Rejected: First object is NOT red")
    elif not result['obj1_is_circle']:
        print(f"   ✗ Rejected: Second object is NOT circle")

if len(rejected_seqs) > 20:
    print(f"\n... and {len(rejected_seqs) - 20} more rejected sequences")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Expected rule: First object is RED AND Second object is CIRCLE")
print(f"Accepted: {len(accepted_seqs)}/{len(test_SIG.sequences)} sequences")
print(f"Rejected: {len(rejected_seqs)}/{len(test_SIG.sequences)} sequences")
if not mismatches:
    print("✓ Rule verification: PASSED")
else:
    print(f"✗ Rule verification: FAILED ({len(mismatches)} mismatches)")
