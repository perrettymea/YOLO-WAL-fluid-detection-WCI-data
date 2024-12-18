import os
import argparse
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
from os import path
import netCDF4 as nc
from tqdm import tqdm



def draw_text(img: object, text: str,
              font: object = cv2.FONT_HERSHEY_DUPLEX,
              pos: tuple = (0, 0),
              font_scale: int = 2,
              font_thickness: int = 1,
              text_color: tuple = (255, 255, 255),
              text_color_bg: tuple = (0, 0, 0)
              ) -> object:
    """
    Draw text on an image with a background rectangle.

    This function draws text on an image with a background rectangle. It is particularly
    useful for adding labels to images, such as those processed with YOLOv5 for object detection.

    Parameters:
    -----------
    img : object
        The input image on which to draw the text. Expected to be a NumPy array
        compatible with OpenCV operations.

    text : str
        The text string to be drawn on the image.

    font : object, optional
        The font type to be used for the text. Default is cv2.FONT_HERSHEY_DUPLEX.

    pos : tuple, optional
        The position (x, y) where the text will be drawn. Default is (0, 0).

    font_scale : int, optional
        The scale factor that is multiplied by the base font size. Default is 2.

    font_thickness : int, optional
        The thickness of the text. Default is 1.

    text_color : tuple, optional
        The color of the text in BGR format. Default is white (255, 255, 255).

    text_color_bg : tuple, optional
        The color of the background rectangle in BGR format. Default is black (0, 0, 0).

    Returns:
    --------
    object
        A tuple containing the width and height of the text.

    """

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + int(text_w / 2), y + text_h - 3), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h - 10), font, font_scale / 2, text_color, font_thickness)

    return text_size


def creation_folder(name_acquisition: str, FOLDER_RESULT: str) -> None:
    """
       Create result folders for storing images and coordinates.

       This function creates a hierarchical folder structure for organizing results
       from an inference experiment. It creates separate folders for storing images
       with bounding boxes and coordinate data of detections.

       Parameters:
       -----------
       name_acquisition : str
           The name of the inference experiment. This will be used as the name of
           the main folder under which subfolders will be created.

       FOLDER_RESULT : str
           The base directory where the result folders will be created.

       Returns:
       --------
       None

       Notes:
       ------
       - The function creates three levels of folders:
         1. A main folder named after the `name_acquisition`
         2. A subfolder named "boxes_images" for storing images with bounding boxes
         3. A subfolder named "coord_detections_center" for storing coordinate data

       - If any of these folders already exist, the function will not raise an error
         due to the use of `exist_ok=True`.

       - The function uses `os.path.join()` to ensure cross-platform compatibility
         when creating file paths.
    """

    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition), exist_ok=True)
    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition, "boxes_images"), exist_ok=True)
    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition, "coord_detections_center"), exist_ok=True)

def _initmodel(model_folder:Path,confidence_threshold:float,name_model:str):
    """
      Initialize and load a YOLOv5 model.

      This function loads a custom YOLOv5 model from a specified path, sets its
      configuration, and returns the initialized model.

      Parameters:
      -----------
      model_folder : Path
          The path to the directory containing the model file.

      confidence_threshold : float
          The confidence threshold for the model's predictions. This value
          determines the minimum confidence score required for a detection
          to be considered valid.

      name_model : str
          The filename of the model to be loaded.

      Returns:
      --------
      object
          The initialized YOLOv5 model.

      Notes:
      ------
      - The function uses torch.hub.load to load the model from a local source.
      - The model is set to use class-agnostic Non-Maximum Suppression (NMS).
      - The model's confidence threshold is set to the provided value.
      - The function prints "Model loading" to indicate the start of the loading process.
      Raises:
      -------
      This function may raise exceptions related to file I/O or model loading,
      such as FileNotFoundError if the model file doesn't exist or RuntimeError
      if there's an issue with loading the model.
      """
    print("Model loading")
    model = torch.hub.load(os.path.join(os.path.dirname(__file__), 'yolov5'), 'custom',
                           path=os.path.join(model_folder, name_model), force_reload=True,
                           source='local')
    model.agnostic = True  # NMS class-agnostic
    model.conf = confidence_threshold
    return model

def _processfile(model, input_file : Path,  FOLDER_RESULT:Path, name_acquisition: str, size_img: int, dB_min: int,dB_max:int):
    """
        Process a NetCDF file containing sonar data, perform object detection, and save results.

        This function reads a NetCDF file containing sonar data, applies a YOLOv5 model for object detection,
        and saves the results as images and coordinate data.

        Parameters:
        -----------
        model : object
            The initialized YOLOv5 model used for object detection.

        input_file : Path
            The path to the input NetCDF file.

        FOLDER_RESULT : Path
            The base directory where results will be saved.

        name_acquisition : str
            The type of acquisition, used for organizing output folders.

        size_img : int
            The size to which the input image should be resized for model inference.

        dB_min : int
            Min dB value to normalize data

        dB_max : int
            Max dB value to normalize data

        Returns:
        --------
        int
            The number of images processed from the NetCDF file.

        Notes:
        ------
        - The function processes .nc (NetCDF) files containing sonar backscatter data.
        - It normalizes the backscatter data and converts it to an image format.
        - Object detection is performed on each ping's data.
        - Detected objects are drawn on the image and saved.
        - Coordinate data for detections are saved in a text file.
        - The function creates necessary subdirectories for organizing outputs.

        Example:
        --------
        ```python
        from pathlib import Path

        model = _initmodel(...)  # Initialize your model
        input_file = Path("/path/to/sonar_data.nc")
        FOLDER_RESULT = Path("/path/to/results")
        name_acquisition = "sonar_survey_1"
        size_img = 640

        num_processed = _processfile(model, input_file, FOLDER_RESULT, name_acquisition, size_img)
        print(f"Processed {num_processed} images from the NetCDF file.")
        ```

        Raises:
        -------
        This function may raise exceptions related to file I/O, NetCDF operations,
        or image processing. Proper error handling should be implemented when using this function.
        """
    root, extension = os.path.splitext(input_file.name)
    nb_img = 0

    print(f"process file {input_file} to {FOLDER_RESULT}")
    if extension == ".nc":
        # READING THE G3D
        Layer = root[:-4]
        line = nc.Dataset(input_file)
        # Add tqdm progress bar for processing pings
        pbar = tqdm(line.groups.keys(), desc="Processing pings", colour="blue",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}", ncols=100)

        for ping in pbar:
            nb_img += 1
            # Access the group within the dataset
            group = line.groups[ping]
            # Accessing specific variable
            variable_name = 'backscatter_mean'
            variable = group.variables[variable_name]
            # Retrieve variable values
            valeurs_variable = variable[:]
            if len(valeurs_variable) > 0:
                valeurs_variable = np.flipud(valeurs_variable)
                # values in dB min max observed on a file to reproduce the data in dB on a 0 to 255 scale (which will be normalised afterwards...)
                valeurs_variable_normalized = (valeurs_variable - dB_min) / (dB_max - dB_min) * 255
                valeurs_variable_normalized = np.clip(valeurs_variable_normalized, 0, 255).astype(np.uint8)
                valeurs_variable_normalized = np.nan_to_num(valeurs_variable_normalized, nan=0)
                normalized_array = valeurs_variable_normalized.astype(np.uint8)
                img = cv2.merge([normalized_array, normalized_array, normalized_array])
                result = model(img, size=size_img)
                box = result.pandas().xyxy[0]
                box2 = box.loc[np.where(box["class"] == 0)]
                box2 = box2.reset_index(drop=True)
                position = []
                if len(box2) > 0:
                    for i in range(0, len(box2)):
                        position += [int(box2["xmin"][i]), int(box2["xmax"][i])]
                        p1, p2 = (int(box2["xmin"][i]), int(box2["ymin"][i])), (
                            int(box2["xmax"][i]), int(box2["ymax"][i]))
                        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
                        draw_text(img, str(np.round(float(box2["confidence"][i]), 2)), font_scale=1,
                                    pos=(int(box2["xmin"][i]), int(box2["ymin"][i]) - 22),
                                    text_color_bg=(0, 0, 255))
                if len(box2) > 0:
                    path_to_layer = os.path.join(FOLDER_RESULT, name_acquisition, "boxes_images",
                                                    Layer)
                    if not os.path.exists(path_to_layer):
                        os.mkdir(path_to_layer)
                    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img2 = Image.fromarray(img_color)
                    datas = img2.getdata()
                    newData = []
                    for item in datas:
                        if item[0] == 128 and item[1] == 0 and item[2] == 0:
                            newData.append((255, 255, 255))
                        else:
                            newData.append(item)
                    img2.putdata(newData)
                    file_name = "{}{:05d}.png".format(Layer, int(ping))
                    file_path = os.path.join(FOLDER_RESULT, name_acquisition, "boxes_images", Layer,
                                                file_name)
                    img2.save(file_path)
                    lat_bab = group.variables["latitude"][0][0]
                    lat_tri = group.variables["latitude"][0][1]

                    lon_bab = group.variables["longitude"][0][0]
                    lon_tri = group.variables["longitude"][0][1]

                    immersion = group.variables['elevation'][0][0]
                    depth = group.variables['elevation'][1][0]

                    for i in range(0, len(box2)):
                        file_name = "{}position_detection_with_z_center_and_file_ping.txt".format(Layer)
                        file_path = os.path.join(FOLDER_RESULT, name_acquisition,
                                                    "coord_detections_center", file_name)

                        with open(file_path, "a+") as myfile:
                            lond = (((int(box2["xmin"][i]) + int(box2["xmax"][i])) / 2) * (
                                    float(lon_tri) - float(lon_bab))) / \
                                    img_color.shape[1] + float(lon_bab)
                            latd = (((int(box2["xmin"][i]) + int(box2["xmax"][i])) / 2) * (
                                    float(lat_tri) - float(lat_bab))) / \
                                    img_color.shape[1] + float(lat_bab)
                            hmax = (((int(box2["ymin"][i])) * (
                                    float(depth) - float(immersion))) /
                                    img_color.shape[0]) + float(immersion)
                            hmin = (((int(box2["ymax"][i])) * (
                                    float(depth) - float(immersion))) /
                                    img_color.shape[0]) + float(immersion)
                            myfile.write(str(lond) + "," + str(latd) + "," + str(
                                (hmin + hmax) / 2) + "," + Layer + "," + "{:05d}".format(
                                int(ping)) + "," + str(float(box2["xmin"][i])) + "," + str(
                                float(box2["xmax"][i])) + "," + str(float(box2["ymin"][i])) + "," + str(
                                float(box2["ymax"][i])) + "," + str(
                                float(box2["confidence"][i])) + "\n")    
    return nb_img


def model2data(Folder_model: str, FOLDER_PICTURES: str, FOLDER_RESULT:str,name_model: str, name_acquisition: str, size_img: int,
               confidence_threshold: float, dB_min: int,dB_max:int) -> None:
    """
     Perform inference on multiple files in a folder using a specified YOLOv5 model.

     This function initializes a YOLOv5 model and processes all compatible files in a specified folder,
     saving the results in a designated output folder.

     Parameters:
     -----------
     Folder_model : str
         The path to the folder containing the model file.

     FOLDER_PICTURES : str
         The path to the folder containing the input files to be processed.

     FOLDER_RESULT : str
         The path where the inference results will be saved.

     name_model : str
         The filename of the model to be used (e.g., "model.pt").

     name_acquisition : str
         A string identifier for the current inference experiment.

     size_img : int
         The size to which input images will be resized before inference.
         This should match the size used during model training.

     confidence_threshold : float
         The confidence threshold for detections. Detections below this threshold will be discarded.

    dB_min : int
        Min dB value to normalize data

    dB_max : int
        Max dB value to normalize data

     Returns:
     --------
     int
         The total number of images processed across all files.
    """
    model = _initmodel(model_folder=Folder_model,confidence_threshold=confidence_threshold,name_model=name_model)
    nb_img = 0
    print("\nInference parameters:")
    print(f"\033[91mFolder_model\033[0m: {Folder_model}")
    print(f"\033[91mFOLDER_PICTURES\033[0m: {FOLDER_PICTURES}")
    print(f"\033[91mFOLDER_RESULT\033[0m: {FOLDER_RESULT}")
    print(f"\033[91mname_model\033[0m: {name_model}")
    print(f"\033[91mname_acquisition\033[0m: {name_acquisition}")
    print(f"\033[91msize_img\033[0m: {size_img}")
    print(f"\033[91mconfidence_threshold\033[0m: {confidence_threshold}")
    print(f"\033[91mdB_min\033[0m: {dB_min}")
    print(f"\033[91mdB_max\033[0m: {dB_max}\n")
    print("Inference begins")
    for file in os.listdir(FOLDER_PICTURES):
        input_file= Path(FOLDER_PICTURES) / Path(file)
        nb_img+=_processfile(model=model, input_file=input_file, FOLDER_RESULT=Path(FOLDER_RESULT), name_acquisition=name_acquisition, size_img=size_img, dB_min=dB_min,dB_max=dB_max)
    return nb_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference on G3D files")
    parser.add_argument(
        "--G3D",
        default='G3D',
        type=str,
        help="Where are the G3D you want to infer on. If you modify it please put absolute path.",
    )
    parser.add_argument(
        "--results",
        default="RESULTS",
        type=str,
        help="Where you want to save your results. If you modify it please put absolute path.",
    )
    parser.add_argument(
        "--folder_model",
        default="NETWORKS",
        type=str,
        help="Where are your model weights? If you modify it please put absolute path.",
    )
    parser.add_argument(
        "--name_acquisition",
        default="TEST_INFERENCE",
        type=str,
        help="Name of your inference experiment",
    )
    parser.add_argument(
        "--name_model",
        default="training_test_with_G3D.pt",
        type=str,
        help="Name of the model you use. Please put it with the extension .pt",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.2,
        type=float,
        help="threshold used to discriminate detections made by the network",
    )
    parser.add_argument(
        "--size_img",
        default=960,
        type=int,
        help="Img with be resized with this value before inference. Has to be related to the value used for training",
    )

    parser.add_argument(
        "--dB_min",
        default=-50,
        type=int,
        help="Min dB value to normalize data",
    )

    parser.add_argument(
        "--dB_max",
        default=10,
        type=int,
        help="Max dB value to normalize data",
    )

    args = parser.parse_args()


    FOLDER_PICTURES = args.G3D
    FOLDER_RESULT = args.results
    FOLDER_MODEL = args.folder_model
    
    creation_folder(args.name_acquisition, FOLDER_RESULT)
    detection = model2data(Folder_model=FOLDER_MODEL,FOLDER_PICTURES= FOLDER_PICTURES,FOLDER_RESULT=FOLDER_RESULT,name_model= args.name_model,
                           size_img=args.size_img, confidence_threshold=args.confidence_threshold,dB_min=args.dB_min,dB_max=args.dB_max, name_acquisition=args.name_acquisition)
    print("End of inference")
