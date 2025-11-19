import stimuli

# Same setup as EXP_4_Perm.py
R = 2
GENERATION_MODE = "Multiset Permutation"

# Use simplified features like in the test
test_SIG = stimuli.Sigma([["r", "b"], ["c","t"], ["l","s"]], ["fill", "shape", "size"], R, generation_mode = "Multiset Permutation")

# P2 formula
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

print("=" * 100)
print("COMPREHENSIVE TEST: P2 Rule with Sequential Conjunct")
print("=" * 100)
print()

# Create the conjunct with "Seq" type
test_conj = test_SIG.form_conjunct(P2, conjunct_type="Seq")

# Print the formula
print("Formula P2 Definition:")
print("  Object 0 (first object in sequence):")
print("    fill='r': =1  (must be exactly 1)")
print("    fill='b': =0  (must be exactly 0)")
print("    shape: +0    (any value)")
print("    size: +0     (any value)")
print()
print("  Object 1 (second object in sequence):")
print("    fill: +0      (any value)")
print("    shape='c': =1 (must be exactly 1)")
print("    shape='t': =0 (must be exactly 0)")
print("    size: +0     (any value)")
print()
print("Expected Rule: First object is RED, Second object is CIRCLE")
print()
print("=" * 100)
print()

# Test all sequences and organize by acceptance
accepted = []
rejected = []

for seq in test_SIG.sequences:
    accepted_flag = test_conj.accepts(seq)
    
    # Get object features
    obj0 = seq.objects[0]
    obj1 = seq.objects[1]
    
    obj0_features = obj0.summarize()
    obj1_features = obj1.summarize()
    
    obj0_fill = obj0_features.get(('fill', 'r'), 0)
    obj0_shape = obj0_features.get(('shape', 'c'), 0)
    obj1_fill = obj1_features.get(('fill', 'r'), 0)
    obj1_shape = obj1_features.get(('shape', 'c'), 0)
    
    # Determine what the objects are
    obj0_is_red = obj0_fill == 1
    obj1_is_circle = obj1_shape == 1
    
    # Check if it matches expected rule
    matches_rule = obj0_is_red and obj1_is_circle
    
    result = {
        'seq': seq,
        'pid': seq.pid,
        'obj0_features': obj0_features,
        'obj1_features': obj1_features,
        'obj0_is_red': obj0_is_red,
        'obj1_is_circle': obj1_is_circle,
        'matches_rule': matches_rule,
        'accepted': accepted_flag
    }
    
    if accepted_flag:
        accepted.append(result)
    else:
        rejected.append(result)

print(f"Total sequences: {len(test_SIG.sequences)}")
print(f"Accepted: {len(accepted)}")
print(f"Rejected: {len(rejected)}")
print()

# Check for mismatches
mismatches = []
for result in accepted:
    if not result['matches_rule']:
        mismatches.append(result)
for result in rejected:
    if result['matches_rule']:
        mismatches.append(result)

if mismatches:
    print("=" * 100)
    print("⚠️  MISMATCHES FOUND!")
    print("=" * 100)
    for result in mismatches:
        print(f"\nSequence PID: {result['pid']}")
        print(f"  Accepted by formula: {result['accepted']}")
        print(f"  Matches expected rule: {result['matches_rule']}")
        print(f"  First object is red: {result['obj0_is_red']}")
        print(f"  Second object is circle: {result['obj1_is_circle']}")
        print(f"  First object: {result['obj0_features']}")
        print(f"  Second object: {result['obj1_features']}")
else:
    print("=" * 100)
    print("✓ VERIFICATION PASSED: All sequences match expected behavior!")
    print("=" * 100)
    print()

# Show all accepted sequences
print("\n" + "=" * 100)
print("ALL ACCEPTED SEQUENCES (should all have: first=red, second=circle):")
print("=" * 100)
for i, result in enumerate(accepted, 1):
    print(f"\n{i}. Sequence PID: {result['pid']}")
    print(f"   First object: {result['obj0_features']}")
    print(f"   Second object: {result['obj1_features']}")
    print(f"   ✓ First is red: {result['obj0_is_red']}, Second is circle: {result['obj1_is_circle']}")

# Show some rejected sequences
print("\n" + "=" * 100)
print("SAMPLE REJECTED SEQUENCES (should NOT have: first=red AND second=circle):")
print("=" * 100)
for i, result in enumerate(rejected[:10], 1):
    print(f"\n{i}. Sequence PID: {result['pid']}")
    print(f"   First object: {result['obj0_features']}")
    print(f"   Second object: {result['obj1_features']}")
    print(f"   First is red: {result['obj0_is_red']}, Second is circle: {result['obj1_is_circle']}")
    if not result['obj0_is_red']:
        print(f"   ✗ Rejected because first object is NOT red")
    elif not result['obj1_is_circle']:
        print(f"   ✗ Rejected because second object is NOT circle")

