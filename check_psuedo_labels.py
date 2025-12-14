#!/usr/bin/env python3
"""
Quick script to check if your existing pseudo-labels are valid.
Run this to see if you need to regenerate them.
"""
import json
import os

def check_pseudo_labels(filepath="pseudo_labels_pqau_t5.json"):
    """Check if pseudo-labels file exists and is valid"""
    
    print("="*60)
    print("Checking Pseudo-Labels Status")
    print("="*60)
    
    if not os.path.exists(filepath):
        print(f"\n❌ File not found: {filepath}")
        print("\n⚠️  You NEED to run Phase 2 to generate pseudo-labels")
        return False
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not data:
            print(f"\n❌ File is empty")
            print("\n⚠️  You NEED to run Phase 2 to generate pseudo-labels")
            return False
        
        # Check structure
        required_fields = ['pubid', 'question', 'contexts', 'long_answer', 'label', 'explanation']
        sample = data[0]
        
        missing = [f for f in required_fields if f not in sample]
        if missing:
            print(f"\n❌ Missing required fields: {missing}")
            print("\n⚠️  You NEED to regenerate pseudo-labels")
            return False
        
        # Check label validity
        valid_labels = {'yes', 'no', 'maybe'}
        label_counts = {'yes': 0, 'no': 0, 'maybe': 0}
        invalid_count = 0
        
        for item in data:
            label = item.get('label', '').lower()
            if label in valid_labels:
                label_counts[label] += 1
            else:
                invalid_count += 1
        
        # Print results
        print(f"\n✅ Pseudo-labels file is valid!")
        print(f"\nTotal examples: {len(data)}")
        print(f"Label distribution:")
        for label in ['yes', 'no', 'maybe']:
            count = label_counts[label]
            pct = (count / len(data)) * 100
            print(f"  {label:>5}: {count:>6} ({pct:>5.1f}%)")
        
        if invalid_count > 0:
            print(f"\n⚠️  Invalid labels: {invalid_count}")
        
        print(f"\n{'='*60}")
        
        # Recommendation
        if len(data) < 1000:
            print("⚠️  You have < 1000 pseudo-labels.")
            print("   Recommendation: Generate more for better performance")
            print("   Suggested: 5000-10000 examples")
        elif len(data) < 5000:
            print("✅ You have a decent number of pseudo-labels.")
            print("   This should work, but more would be better.")
            print("   Suggested: 10000+ examples for best results")
        else:
            print("✅ You have plenty of pseudo-labels!")
            print("   This is great - you can proceed with Phase 3")
        
        print("\n🎯 Decision: You can SKIP Phase 2 and use existing data")
        print("   Just make sure the file is in the same directory as Phase 3")
        print(f"{'='*60}\n")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")
        print("\n⚠️  You NEED to regenerate pseudo-labels")
        return False
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
        print("\n⚠️  You NEED to regenerate pseudo-labels")
        return False


if __name__ == "__main__":
    import sys
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "pseudo_labels_pqau_t5.json"
    
    check_pseudo_labels(filepath)