# Image Captioning and Face Detection with Voice Commands

This project integrates image captioning, face detection, and voice command recognition. It utilizes Salesforce's BLIP (Bootstrapped Language Image Pretraining) model for generating image captions, OpenCV for face detection, and Python's `speech_recognition` library for voice command processing.

## Installation

Before using the project, the following Python packages need to be installed:

- `torch`
- `transformers`
- `opencv-python`
- `speech_recognition`

You can install these packages using the following pip command:

```bash
pip install torch transformers opencv-python speech_recognition
```

##Usage
To run the program, execute the main Python file (<file_name>.py):

```bash
python <file_name>.py
```
The program captures images from the camera, detects faces, and generates captions for the images. It can also recognize and respond to specific voice commands.

##Voice Commands
The program can recognize the following voice commands:

"Hello": Makes the program greet.
"Camera status": Provides information about the camera's status.
"Help": Lists available commands.
"Close program": Shuts down the program.
"Open Chrome": Starts Google Chrome (requires additional functionality).
"What do you see": Displays the caption of the image.

##License
This project is licensed under the MIT License.


This Markdown content provides a comprehensive guide for your README file, covering installation, usage, voice commands, and licensing information for your project.
