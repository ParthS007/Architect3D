## Evaluation Baseline for OpenArchitect3D

1. Mask3D with ScanNet200
2. OpenMask3D with ScanNet200
2. OpenMask3D with ScanNet++


### 3D instance segmentation results

| Model                                                     | Image Features |  AP  | AP50 | AP25 | head (AP)  | common (AP)  | tail (AP)  |
|-----------------------------------------------------------|----------------|------|------|------|------------|--------------|------------|
| **Closed-vocabulary, fully supervised**                   |                |      |      |      |            |              |            |
| Mask3D                                                    | -              | 26.9 | 36.2 | 41.4 | 39.8       | 21.7         | 17.9       |
| **Open-vocabulary**                                       |                |      |      |      |            |              |            |
| OpenMask3D-ScanNet200                                     | CLIP           | 15.4 | 19.9 | 23.1 | 17.1       | 14.1         | 14.9       |
| OpenArchirect3D-ScanNet++ (Ours)                          | CLIP           |      |      |      |            |              |            |

## Generalisation Results

| Method                                    | Mask Training  | Novel Classes          ||| Base Classes          ||| All Classes | tail (AP) |
|-------------------------------------------|----------------|------------------------|----|------|------------------------|----|------|-------------|-----------|
|                                           |                | AP     | AP50  | AP25  | AP     | AP50  | AP25  | AP          |           |
| OpenMask3D-ScanNet20                      | ScanNet20      | 11.9   | 15.2  | 17.8  | 14.3   | 18.3  | 21.2  | 12.6        | 11.5      |
| OpenMask3D-ScanNet200                     | ScanNet200     | 15.0   | 19.7  | 23.1  | 16.2   | 20.6  | 23.1  | 15.4        | 14.9      |
| OpenArchirect3D-ScanNet++ (Ours)          | ScanNet++      |        |       |       |        |       |       |             |           |

## Results on Architectural Elements

| Method                                    | Mask Training  | Novel Classes          ||| Base Classes          ||| All Classes | tail (AP) |
|-------------------------------------------|----------------|------------------------|----|------|------------------------|----|------|-------------|-----------|
|                                           |                | AP     | AP50  | AP25  | AP     | AP50  | AP25  | AP          |           |
| OpenMask3D                                | ScanNet200     |        |       |       |        |       |       |             |           |
| OpenArchirect3D           (Ours)          | ScanNet++      |        |       |       |        |       |       |             |           |
