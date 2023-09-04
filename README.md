# Prune and Merge for Segmenter 

The implementation of Prune and Merge Method for Segmenter.
 
## Installation

The installation can follow the orginal segmenter [link]()

## Model Zoo
We release models of compressed Seg-L-Mask/16 with different compression rate.

### ADE20K

<table>
  <tr>
    <th>Name</th>
    <th>mIoU (SS)</th>
    <th>Gflops </th>
    <th>Resolution</th>
    <th>FPS</th>
    <th colspan="3">Download</th>
  </tr>
<tr>
    <td>Seg-L-Mask/16</td>
    <td>51.8</td>
    <td>658</td>
    <td>640x640</td>
    <td>4.18</td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/checkpoint.pth">model</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/variant.yml">config</a></td>
  </tr>
<tr>
    <td>Seg-L-R08</td>
    <td>51.9</td>
    <td>528</td>
    <td>640x640</td>
    <td>5.25</td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_small_mask/checkpoint.pth">model</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/variant.yml">config</a></td>
  </tr>
<tr>
    <td>Seg-L-R07</td>
    <td>51.7</td>
    <td>456</td>
    <td>640x640</td>
    <td>5.74</td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_mask/checkpoint.pth">model</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/variant.yml">config</a></td>
  </tr>
<tr>
    <td>Seg-L-R05</td>
    <td>50.2</td>
    <td>364</td>
    <td>640x640</td>
    <td>6.85</td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_patch8/checkpoint.pth">model</a></td>
    <td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/variant.yml">config</a></td>
  </tr>
</table>



## Inference

Download one checkpoint with its configuration in a common folder, for example `seg_large_mask`.

To evaluate on ADE20K, run the command:
```python
# single-scale evaluation:
python -m segm.eval.miou seg_large_mask/checkpoint.pth ade20k --singlescale
# multi-scale evaluation:
python -m segm.eval.miou seg_large_mask/checkpoint.pth ade20k --multiscale
```

## Compress

Compress `Seg-Large-Mask/16` on ADE20K on 2 3090 GPU:
(You should change the batch size in config.yml according to your GPU number)
```python
python -m torch.distributed.launch --master_port=3001 --nproc_per_node=2 --use_env \
prune_finetune.py --prune --pretrain-dir path/to/pretrain/dir \
--dataset ade20k --log-dir path/to/log/dir \
--backbone vit_large_patch16_384 --decoder mask_transformer --iter-num 100
```

## Finetune

Finetune `Seg-Large-Mask/16` on ADE20K on 8 3090 GPU:
```python
python -m torch.distributed.launch --master_port=3001 --nproc_per_node=8 --use_env \
prune_finetune.py --eval --pretrain-dir path/to/pretrain/dir \
--dataset ade20k --log-dir path/to/log/dir \
--distill-type soft --teacher-path path/to/pretrain/model \
--backbone vit_large_patch16_384 --decoder mask_transformer --iter-num 0  \
--scheduler cosine -lr 1e-4 --alpha 1.0 --tau 20 --weight-decay 0.0001 
```




## Acknowledgements

Our implementation is base on [segmenter]()
