import stimuli

# Same setup
R = 2
GENERATION_MODE = "Multiset Permutation"

test_SIG = stimuli.Sigma([["r", "b"], ["c","t"], ["l","s"]], ["fill", "shape", "size"], R, generation_mode = "Multiset Permutation")

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
print("COMPARING: P2 with 'Seq' vs 'Product' conjunct_type")
print("=" * 100)
print()

# Test with "Seq"
test_conj_seq = test_SIG.form_conjunct(P2, conjunct_type="Seq")
print("With conjunct_type='Seq' (order matters):")
print("  Rule: First object is RED AND Second object is CIRCLE")
print()

# Test with "Product" 
test_conj_product = test_SIG.form_conjunct(P2, conjunct_type="Product")
print("With conjunct_type='Product' (order doesn't matter):")
print("  Rule: Sequence contains a RED object AND a CIRCLE object (anywhere)")
print()

# Find a sequence that's True with Product but False with Seq
# Example: red circle + red triangle
# - With Seq: False (second is triangle, not circle)
# - With Product: True (has red object AND circle object)

print("=" * 100)
print("Testing specific sequence: red circle + red triangle")
print("=" * 100)

# Find sequences with red circle first, red triangle second
for seq in test_SIG.sequences:
    summary = seq.summarize()
    obj0 = summary[0]
    obj1 = summary[1]
    
    # Check if it's red circle + red triangle
    obj0_is_red_circle = obj0.get(('fill', 'r'), 0) == 1 and obj0.get(('shape', 'c'), 0) == 1
    obj1_is_red_triangle = obj1.get(('fill', 'r'), 0) == 1 and obj1.get(('shape', 't'), 0) == 1
    
    if obj0_is_red_circle and obj1_is_red_triangle:
        accepts_seq = test_conj_seq.accepts(seq)
        accepts_product = test_conj_product.accepts(seq)
        
        print(f"\nSequence PID: {seq.pid}")
        print(f"  Objects: {summary}")
        print(f"  First: red circle, Second: red triangle")
        print(f"  With 'Seq': {accepts_seq} (should be False - second is not circle)")
        print(f"  With 'Product': {accepts_product} (should be True - has red AND circle)")
        
        if accepts_product and not accepts_seq:
            print(f"\n  ⚠️  THIS IS THE ISSUE!")
            print(f"     With 'Product', this sequence is True")
            print(f"     With 'Seq', this sequence is False")
            print(f"     Your experiment uses 'Product', so this would show as True!")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print("Your test (line 106) uses: conjunct_type='Seq'")
print("Your experiment (line 130) uses: conjunct_type='Product'")
print()
print("SOLUTION: Change line 130 to use 'Seq' instead of 'Product'")
print("  Change: prod_conj = SIG.form_conjunct(P2, conjunct_type = 'Product')")
print("  To:     prod_conj = SIG.form_conjunct(P2, conjunct_type = 'Seq')")

