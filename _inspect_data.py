"""Inspect data/{traindata,testdata} structure in detail."""
import os
from pathlib import Path
from collections import Counter, defaultdict

os.chdir('/root/CFANet-main')

for split in ['traindata', 'testdata']:
    root = f'data/{split}'
    patients = sorted(os.listdir(root))
    print(f'\n=== {split} : {len(patients)} patients ===')

    n_with_t1 = n_with_t2 = n_with_t1_mask = n_with_t2_mask = 0
    n_paired = 0
    depth_counter = Counter()
    file_count_per_patient = []
    missing = []
    no_t1_image = no_t2_image = no_t1_mask = no_t2_mask = 0
    skip = 0

    for p in patients:
        pdir = Path(root) / p
        t1dir = pdir / 't1'
        t2dir = pdir / 't2'

        if not t1dir.is_dir() or not t2dir.is_dir():
            missing.append(p)
            continue
        skip += 1

        # find T1 image and mask
        t1_files = sorted([f for f in t1dir.iterdir() if f.suffix == '.gz'])
        t2_files = sorted([f for f in t2dir.iterdir() if f.suffix == '.gz'])

        t1_img = [f for f in t1_files if 'label' not in f.name.lower() and 'mask' not in f.name.lower()]
        t1_msk = [f for f in t1_files if 'label' in f.name.lower() or 'mask' in f.name.lower()]
        t2_img = [f for f in t2_files if 'label' not in f.name.lower() and 'mask' not in f.name.lower()]
        t2_msk = [f for f in t2_files if 'label' in f.name.lower() or 'mask' in f.name.lower()]

        if not t1_img: no_t1_image += 1
        if not t2_img: no_t2_image += 1
        if not t1_msk: no_t1_mask += 1
        if not t2_msk: no_t2_mask += 1
        if t1_img and t2_img and t1_msk and t2_msk:
            n_paired += 1

        if t1_img: n_with_t1 += 1
        if t2_img: n_with_t2 += 1
        if t1_msk: n_with_t1_mask += 1
        if t2_msk: n_with_t2_mask += 1
        file_count_per_patient.append(len(t1_files) + len(t2_files))

    print(f'  with_t1_image    : {n_with_t1}/{len(patients)}')
    print(f'  with_t2_image    : {n_with_t2}/{len(patients)}')
    print(f'  with_t1_mask     : {n_with_t1_mask}/{len(patients)}')
    print(f'  with_t2_mask     : {n_with_t2_mask}/{len(patients)}')
    print(f'  FULLY PAIRED     : {n_paired}/{len(patients)}  (have T1+T2 img+mask)')
    print(f'  file cnt (per patient) min/median/max: '
          f'{min(file_count_per_patient)}/{sorted(file_count_per_patient)[len(file_count_per_patient)//2]}/{max(file_count_per_patient)}')
    if missing:
        print(f'  missing t1/t2 dirs: {len(missing)}, examples: {missing[:3]}')

# === quick depth check on a sample ===
print('\n=== sample volume shapes (first 3 patients of testdata) ===')
import SimpleITK as sitk
for p in sorted(os.listdir('data/testdata'))[:3]:
    pdir = Path('data/testdata') / p
    for mod in ['t1', 't2']:
        for f in (pdir / mod).iterdir():
            if 'label' in f.name.lower() or 'mask' in f.name.lower():
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
            print(f'  {p}/{mod}/{f.name}: shape={arr.shape}, '
                  f'range=[{arr.min():.1f}, {arr.max():.1f}]')
