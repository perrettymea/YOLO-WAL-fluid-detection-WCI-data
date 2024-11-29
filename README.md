# *WAL*: Fluid emission detection- by *W*ater-column *A*coustics and deep *L*earning-approach

<div align="center">
<table>
  <tr>
    <td><img src="IMG/LOGO/Image2.jpg" alt="ifremer"></td>
    <td><img src="IMG/LOGO/Logo-Region-Bretagne_sombre.jpg" alt="Image 2"></td>
    <td><img src="IMG/LOGO/ENSTABretagne-LogoH-RVB-COULEUR.jpg" alt="Image 3"></td>
  </tr>
</table>
</div>




Detecting and locating emitted fluids in the water column is necessary for studying margins, identifying natural resources, and preventing geohazards. Fluids can be detected in the water column using multibeam echosounder data. 

<div align="center">
<table>
  <tr>
    <td><img src="IMG/LOGO/couv.PNG" alt="COUV" width="400" height="400"></td>
  </tr>
</table>
</div>

However, manually analyzing the huge volume of this data for geoscientists is a very time-consuming task. Our study investigated the use of a YOLO-based deep learning supervised approach to automate the detection of fluids emitted from cold seeps (gaseous methane) and volcanic sites (liquid carbon dioxide). Several thousand annotated echograms collected from different seas and oceans during distinct surveys were used to train and test the deep learning model. Additionally, we thoroughly analyzed the composition of the training dataset and evaluated the detection performance based on various training configurations. The tests were conducted on a dataset comprising hundreds of thousands of echograms i) acquired with three different multibeam echosounders (Kongsberg EM302 and EM122 and Reson Seabat 7150) and ii) characterized by variable water column noise conditions related to sounder artefacts and the presence of biomass (fishes, dolphins). 


This repository contains the code for inference with YOLOv5 and models trained for fluid emission detection on various Multibeam Echosounders (EM122, EM302, Reson Seabat 7150). This fluid detector was already used for near-real time acquisition detection during the cruises  [MAYOBS23 (EM122 - 2022)]([URL](https://campagnes.flotteoceanographique.fr/campaign?id=18002494)) and  [HAITI-TWIST (Seabat Reson 7150 - 2024)]([URL](https://campagnes.flotteoceanographique.fr/campagnes/18001258)).

Weights of the neural networks are available on the following doi:






## Table of Contents




## How to install YOLOv5-WAL

Here's how to install the environment. 

```
git clone https://github.com/this_code
conda create -n YOLOV5G3D
conda activate YOLOV5G3D
```

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
```
Then:

```
cd yolov5
pip install -r requirements.txt
pip install netCDF4 #to read g3D file
```

## How to perform an inference on multi-beam data with GLOBE

Your MBES data is in raw format (.all/.wcd, .kmall, .s7k...), so you'll need to convert it to get a cartesian representation of each ping:


**Manual method**


Converting the raw file into a g3D file:

* Convert your raw file in XSF (following the SONAR-netcf4 convention for sonar data).
<div align="center">
<table>
  <tr>
    <td><img src="IMG\SCREENSHOTS\2024-11-29 13_55_45-Globe.png" alt="export_xsf" ></td>
  </tr>
</table>
</div>

* Our file is not already in cartesian representation so we to need to convert it in G3D netcdf format.

<div align="center">
<table>
  <tr>
    <td><img src="IMG\SCREENSHOTS\2024-11-29 13_56_44-Globe.png" alt="export_g3D" ></td>
  </tr>
</table>
</div>

This G3D contains the following informations that you can access:


```
Groups:
  Group: [Ping number]

  Variables:
        elevation: ('vector', 'position') float32
          Attributes:
            units: meters
            long_name: elevation
            standard_name: elevation
        longitude: ('vector', 'position') float64
          Attributes:
            units: degrees_east
            long_name: longitude
            standard_name: longitude
        latitude: ('vector', 'position') float64
          Attributes:
            units: degrees_north
            long_name: latitude
            standard_name: latitude
        backscatter_mean: ('height', 'length') float32
          Attributes:
            units: dB
            long_name: backscatter_mean
            standard_name: backscatter_mean
```

This manual method must be used for all raw files before inference. Coming soon **Robot/automatic method to automatically infer on raw data**




If you have other software/code that can extract pings from the water column and represent it as a 2D-cartesian-matrix format (numpy, as with g3D), you can direct it to the neural network for inference. As neural networks were not trained on our specific format, be careful to fit with g3D outputs.

## Parameters to be set for the inference


* *G3D*: Path to the folder containing G3D files for inference (default: 'G3D')
* *results*: Path to save inference results (default: 'RESULTS')
* *folder_model*: Path to the folder containing model weights (default: 'NETWORKS')
* *name_acquisition*: Name of the inference experiment (default: 'TEST_INFERENCE')
* *name_model*: Name of the model file to use, including .pt extension (default: 'training_test_with_G3D.pt')
* *confidence_threshold*: Threshold for discriminating detections (default: 0.3)
* *size_img*: Size to resize images before inference, has to be a multiple of 32 as detailed in YOLOv5 documentation (will be resized if not) (default: 960)
* *dB_min*: Minimum dB value for data normalization (default: -50)
* *dB_max*: Maximum dB value for data normalization (default: 10)

In the event that the dB_min/dB_max values are not adequately defined, the resulting inference will be of poor quality. This is due to the fact that the discrepancy between the features of the training and inference data will be too significant. You have to fix these limits in order to see properly your fluid echoes.

For more documentation YOLOv5 training see : [YOLOv5 documentation](https://github.com/ultralytics/yolov5)


## Example Usage

```
python inference_on_G3D.py  --name_acquisition TEST --confidence_threshold 0.3 --name_model PAMELA_MOZ1_EM122_EM302_Reson_Seabat.pt
```

**Results:**

<div align="center">
<table>
  <tr>
    <td><img src="IMG\SCREENSHOTS\terminal.JPG" alt="terminal" ></td>
  </tr>
</table>
</div>

Folders are created, one with the images on which the detections are made and one with the coordinates of the detections, sorted by the original G3D file.

<div align="center">
<table>
  <tr>
    <td><img src="IMG\SCREENSHOTS\example_detection.png" alt="detection" ></td>
  </tr>
</table>
</div>


The coordinates of the detections can be used for visualisation in a Geographic Information System and correspond to the mid-point of the detection box. 
The following parameters were recorded for each detection:

| **Parameter**                     | **Description**                                          |
|-----------------------------------|---------------------------------------------------------|
| **Longitude (WGS84)**             | Longitude of the detected object (center of the box).                       |
| **Latitude (WGS84)**                      | Latitude of the detected object (center of the box).                        |
| **Average Height**               | Average height calculated as \((h_{\text{min box}} + h_{\text{max box}}) / 2\). |
| **File Name**                    | Name of the file where the detection occurred.         |
| **Ping**                         | The specific ping number associated with the detection. |
| **Box Coordinates**              | Coordinates of the bounding box in the image (in pixels).          |
| **Confidence Index**             | Confidence score of the detection from the model.      |







For more details please refer to the following resources:
* Article link
* [YOLOv5 documentation]([URL](https://github.com))
* [GLOBE](https://www.seanoe.org/data/00592/70460/)

## Contact
For questions or support, please contact tymea.perret@ifremer.fr.
