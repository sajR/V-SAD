# Feture Extraction
The networks utilise mouth-region images to detect speech. This is done so less noise is introduced to the network which can be the case of using full-face images.

Extraction of mouth-region images is done by using Haar Cascade and heuristics. The Haar cascade identifies the face in the image, once found calculations are done to obtain a mouth-region image.

The images provided by mouth detection can be inconsistent due to lightning conditions, race/colour and beards etc. As a result, images are enhanced by controlling the brightness and contrast of the image. This results in consistent images throughout the dataset. 
