# Pytorch implementation of "MSPD-Net Multi-Scale and Structural-Appearance Prototype Decoupled Network for Weakly Supervised Semantic Segmentation" (Under Review).


### Installation  ###

Install dependencies:
```
pip install -r requirements.txt
```
### Data Preparation 
<details>
<summary>
PASCAL VOC 2012
</summary>

- Download [the PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).
  ``` bash
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  tar –xvf VOCtrainval_11-May-2012.tar
  ```
- Download augmented annoations `SegmentationClassAug.zip` from [SBD dataset] via this [link](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).
- Make your data directory like this below
  ``` bash
  VOCdevkit/
  └── VOC2012
      ├── Annotations
      ├── ImageSets
      ├── JPEGImages
      ├── SegmentationClass
      ├── SegmentationClassAug
      └── SegmentationObject
    ```

  </details>

  <details>
  <summary>
  MS COCO 2014
  </summary>
  
  - Download [MS COCO 2014 dataset](https://cocodataset.org/#home)
    ``` bash
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
    ```
    </details>

### Training ###

Training on VOC:
```
bash train/run_voc.sh
```
Training on COCO:

```
bash train/run_voc.sh
```




