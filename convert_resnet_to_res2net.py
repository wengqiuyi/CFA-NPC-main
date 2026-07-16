"""
Helper: convert a ResNet-50 state-dict into a Res2Net-50 state-dict
that can be loaded into the CFANet backbone(s).

NOTE: Res2Net-50 uses Bottle2neck blocks whose internal structure is
DIFFERENT from ResNet-50's Bottleneck blocks. The two share the
following components that can be transferred 1:1:

    ResNet-50                 Res2Net-50
    --------------------------------------------------
    conv1 (7x7)        <--->  backbone.conv1
    bn1                <--->  backbone.bn1
    layer{i}.0.conv1   <--->  layer{i}.0.conv1   (1x1 reduce)
    layer{i}.0.bn1     <--->  layer{i}.0.bn1
    layer{i}.0.conv3   <--->  layer{i}.0.conv3   (1x1 expand)
    layer{i}.0.bn3     <--->  layer{i}.0.bn3
    layer{i}.0.downsample.{0,1,2}  (only when stride>1)
    layer{i}.j.conv1   (1x1 reduce, j>=1)
    layer{i}.j.bn1
    layer{i}.j.conv3   (1x1 expand, j>=1)
    layer{i}.j.bn3

The 3x3 middle convs of ResNet (conv2 / bn2) DO NOT exist in
Res2Net's hierarchical convs and are skipped.

Usage:
    python convert_resnet_to_res2net.py \\
        --src   ./pretrain/resnet_50_23dataset.pth \\
        --dst   ./lib/resnet50_to_res2net.pth
"""
import argparse
import os
import torch


def convert(src_state):
    """Map a ResNet-50 state-dict to a Res2Net-50 state-dict."""
    new_state = {}
    skipped = []

    for k, v in src_state.items():
        # ----- stem -----
        if k.startswith('conv1.') or k == 'conv1.weight':
            new_state['conv1.' + k.split('.', 1)[1] if '.' in k else 'conv1.weight'] = v
            continue
        if k.startswith('bn1.'):
            new_state['bn1.' + k.split('.', 1)[1]] = v
            continue

        # ----- residual stages -----
        # ResNet-50 keys:  layer1.0.conv1.weight, layer1.1.conv2.weight, ...
        if k.startswith('layer'):
            # Skip the 3x3 conv and its BN of the Bottleneck (Res2Net uses
            # hierarchical convs instead). Keep conv1 (1x1), bn1, conv3 (1x1), bn3.
            tail = k.split('.', 2)[2]                # e.g. '0.conv2.weight'
            if tail.startswith('conv2') or tail.startswith('bn2'):
                skipped.append(k)
                continue
            new_state[k] = v

    return new_state, skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='ResNet-50 .pth')
    p.add_argument('--dst', required=True, help='output Res2Net-50 .pth')
    args = p.parse_args()

    print('Loading {} ...'.format(args.src))
    src = torch.load(args.src, map_location='cpu')
    if isinstance(src, dict) and 'state_dict' in src:
        src = src['state_dict']

    new_state, skipped = convert(src)
    print('  transferred : {} tensors'.format(len(new_state)))
    print('  skipped     : {} tensors (3x3 conv/bn)'.format(len(skipped)))

    torch.save(new_state, args.dst)
    print('Saved -> {}'.format(args.dst))
    print()
    print('Now train with:')
    print('  python train.py --pretrain_ckpt {} --load_backbone_only'.format(args.dst))


if __name__ == '__main__':
    main()
