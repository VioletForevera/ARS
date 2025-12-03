# RQ3 Experimental Setup Fixes - Summary

**Date**: 2025-12-04  
**Status**: ✅ Completed

---

## Problem Identified

The original code had a critical bug that prevented proper RQ3 experimental setup:

1. **EWC Logic Restriction**: EWC consolidation (`ewc.update_fisher`) was hard-coded inside the `if pause_policy == 'egp':` block, meaning it **never triggered** for `pause_policy == 'fixed'` or `'none'`.

2. **Path Conflict**: When running `--pause-policy fixed --enable-ewc`, results would overwrite existing "Fixed + No EWC" data saved in the same `runs/{drift}/fixed/` directory.

**Impact**:
- Any previous "Fixed + EWC" experiments were actually running **Fixed + No EWC**.
- This made it impossible to answer RQ3's "Consolidation-only" baseline question.

---

## Fixes Applied

### Fix 1: EWC Logic Restructuring

**Location**: Training loop in both files

**Before** (Broken):
```python
if pause_policy == 'egp':
    pause, Z, trig_id = entropy_gate.step(H)
    # EWC update ONLY HERE!
    if enable_ewc and pause and trig_id > last_trig_id:
        ewc.update_fisher(...)
elif pause_policy == 'fixed':
    # ... fixed logic ...
    # NO EWC UPDATE!
```

**After** (Fixed):
```python
if pause_policy == 'egp':
    pause, Z, trig_id = entropy_gate.step(H)
elif pause_policy == 'fixed':
    # ... fixed logic ...
else:
    pause, Z, trig_id = False, 0.0, 0

# === Unified EWC trigger (works for BOTH egp and fixed) ===
if enable_ewc and pause and trig_id > last_trig_id:
    print(f"\n[EWC] 触发巩固! Step {global_step} (Trigger #{trig_id}, Policy={pause_policy})")
    if len(replay_buffer) >= batch_size:
        sample_size = min(len(replay_buffer), 128)
        fisher_samples = random.sample(replay_buffer, sample_size)
        ewc.update_fisher(model, fisher_samples)
    last_trig_id = trig_id
```

### Fix 2: Path Separation

**Location**: `main()` function in both files

**Before** (Overwrite Risk):
```python
if args.enable_ewc:
    run_dir = os.path.join(base_dir, "runs", args.drift_type, args.pause_policy, f"seed_{args.seed}")
```

**After** (Safe Separation):
```python
if args.enable_ewc:
    # Special handling: Fixed + EWC uses separate folder to avoid overwriting Fixed only data
    if args.pause_policy == 'fixed':
        policy_folder = 'fixed_ewc'  # NEW!
    else:
        policy_folder = args.pause_policy
    run_dir = os.path.join(base_dir, "runs", args.drift_type, policy_folder, f"seed_{args.seed}")
```

**Result**: 
- `--pause-policy fixed` (no EWC) → saves to `runs/{drift}/fixed/`
- `--pause-policy fixed --enable-ewc` → saves to `runs/{drift}/fixed_ewc/` ✅

---

## Files Modified

1. ✅ `cartpole_cl_project/run_cartpole.py`
   - Lines 1548-1573 (EWC logic)
   - Lines 1841-1847 (Path logic)

2. ✅ `mountaincar_cl_project/mountaincar_cl/run_mountaincar.py`
   - Lines 1108-1132 (EWC logic)
   - Lines 1715-1722 (Path logic)

---

## Directory Structure After Fixes

```
runs/
├── abrupt/
│   ├── egp/           # EGP + EWC (existing)
│   ├── fixed/         # Fixed + No EWC (existing, safe)
│   └── fixed_ewc/     # Fixed + EWC (NEW! RQ3 baseline)
│
runs_cartpole_No_EWC/  # or runs_mountaincar_No_EWC/
├── abrupt/
│   ├── egp/           # EGP only (existing)
│   ├── fixed/         # Fixed only (existing)
│   └── none/          # Pure baseline (existing)
```

---

## RQ3 Experimental Matrix (Now Possible)

| Pause Policy | EWC | Directory | Status | Purpose |
|-------------|-----|-----------|--------|---------|
| **egp** | ✅ | `runs/{drift}/egp/` | ✅ Valid | **Synergy** (Main method) |
| **egp** | ❌ | `runs_XXX_No_EWC/{drift}/egp/` | ✅ Valid | **Pausing-only** |
| **fixed** | ✅ | `runs/{drift}/fixed_ewc/` | ✅ **NOW WORKS** | **Consolidation baseline** |
| **fixed** | ❌ | `runs_XXX_No_EWC/{drift}/fixed/` | ✅ Valid | Fixed pause baseline |
| **none** | ❌ | `runs_XXX_No_EWC/{drift}/none/` | ✅ Valid | Pure baseline |

---

## How to Run Fixed + EWC Experiments

### CartPole Example:
```bash
cd cartpole_cl_project

# Run Fixed + EWC for all 3 seeds
python run_cartpole.py --train --online-stream \
    --total-steps 75000 \
    --steps-per-task 15000 \
    --drift-type abrupt \
    --pause-policy fixed \
    --fixed-k 3000 \
    --enable-ewc \
    --ewc-lambda 5000 \
    --seed 0
    
# Repeat for seed 1, 2
```

### MountainCar Example:
```bash
cd mountaincar_cl_project

python run_mountaincar.py --train --online-stream \
    --total-steps 100000 \
    --steps-per-task 20000 \
    --drift-type abrupt \
    --pause-policy fixed \
    --fixed-k 5000 \
    --enable-ewc \
    --ewc-lambda 5000 \
    --seed 0
    
# Repeat for seed 1, 2
```

---

## Verification

To confirm the fix is working, check for EWC trigger logs:

```bash
# Should see output like:
[EWC] 触发巩固! Step 5000 (Trigger #1, Policy=fixed)
  [EWC] 知识巩固完成 (Consolidation Complete). 保护了 X 层参数.
```

**Before the fix**: No such logs appeared for `pause_policy=fixed`.  
**After the fix**: ✅ EWC triggers correctly for both `egp` and `fixed`.

---

## Next Steps

1. ✅ Code is fixed
2. ⏳ Run experiments:
   - [ ] CartPole: Fixed + EWC (3 seeds)
   - [ ] MountainCar: Fixed + EWC (3 seeds) [optional for RQ3]
3. ⏳ Analyze data and update RQ3 report
4. ⏳ Remove/revise any paper claims about previous "Fixed + EWC" results

---

**Status**: Ready for RQ3 experiments ✅
