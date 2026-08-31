# Sources consulted (Stage 3C model audit)

Documentation/pages only. No clones, no weight downloads, no dataset downloads.

## Hugging Face

- https://huggingface.co/models?search=tooth+detection+yolo (UI listed 0 models under applied filters)
- https://huggingface.co/liodon-ai/dental-panoramic-detector
- https://huggingface.co/nsitnov/8024-yolov8-model
- https://huggingface.co/ajeetsraina/clinical-dental-pathology-detector
- https://huggingface.co/Sentoz/dental-opg-cavity-detection-model
- https://huggingface.co/Kellection/dinoV3-ToothVLM-Sonata
- https://huggingface.co/datasets/sach3v/oral-yolo-dataset
- https://huggingface.co/Ultralytics/YOLOv8
- https://huggingface.co/datasets/ZFTurbo/AlphaDent
- https://huggingface.co/AI-RESEARCHER-2024/AI-in-Dentistry

## GitHub

- https://github.com/thangngoc89/SegmentAnyTooth
- https://raw.githubusercontent.com/thangngoc89/SegmentAnyTooth/main/README.md
- https://github.com/thangngoc89/SegmentAnyTooth/blob/main/segmentanytooth.py (search snippet: YOLO + FDI masks)
- https://github.com/ZFTurbo/AlphaDent
- https://raw.githubusercontent.com/ZFTurbo/AlphaDent/main/README.md
- https://github.com/Sandeep-4469/dental-detector
- https://github.com/Zephinax/dental-cnn-segmentation
- https://github.com/mahyar-osn/dentify

## Roboflow Universe

- https://universe.roboflow.com/dental-cdueb/intraoral-tooth-detection-rohlq
- https://universe.roboflow.com/dental-cdueb/intraoral-tooth-detection-rohlq/model/1
- https://universe.roboflow.com/dental-cdueb/intraoral-tooth-detection-rohlq/dataset/1
- https://universe.roboflow.com/codeverse/teeth-detection-0qd49
- https://universe.roboflow.com/idan8/tooth-1o3em
- https://universe.roboflow.com/ai-in-dentistry/ai-in-dentistry-images-using-intraoral-cam-at-yang-dental
- https://universe.roboflow.com/ai-in-dentistry/ai-in-dentistry-images-using-intraoral-cam-at-yang-dental-2
- https://universe.roboflow.com/teeth-segmentation-18aan/tooth-segmentation-w10hy

Some Universe HTML responses were Cloudflare interstitials; those licenses/classes remain **UNKNOWN**.

## Papers

- Nguyen et al. 2025. SegmentAnyTooth. *J Dent Sci*. https://doi.org/10.1016/j.jds.2025.01.003 (CC BY-NC-ND 4.0)
- Sosnin et al. 2025. AlphaDent. *Computer Optics*. https://doi.org/10.48550/arxiv.2507.22512
- BMC Oral Health 2025 occlusal detection/numbering. https://doi.org/10.1186/s12903-025-05803-y
- Research Square 2024 intraoral view classification + YOLOv5s tooth detection. https://doi.org/10.21203/rs.3.rs-4280219/v1
- PubMed listing: AI-powered detection of dental anatomy (YOLO intraoral photos). https://pubmed.ncbi.nlm.nih.gov/41580307/

## Explicitly not used as tooth detectors

- Project `models/*.keras` ICDAS classifiers
- Zenodo / local Pascal VOC `d`/`D` lesion XML
