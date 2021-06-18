# Voyage
Scripts used for lane detection while driving


<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://user-images.githubusercontent.com/27528504/120189327-7e10d200-c217-11eb-9c1c-1f7721fb2d4f.png" alt="Logo" width="400" height="150">
  </a>

  <h1 align="center">Real time lane detection for smart car driving</h1>


Applying Python and Opencv for lane detection while driving

In Lanes.py,  is a script where opencv was deployed to preprocess each frame and detect lanes on the road. You can test the result on the cars2.mp4.
Here is a short presentation of this project: https://docs.google.com/presentation/d/1bpDDgozkksV3VoO_ukPrCPahzJl7FlEJeR93rsXWnNM/edit?usp=sharing



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#goal-of-this-project">Goal of this project</a>
    </li>
    <li>
      <a href="#installation-and-system-setup">Installation and system setup</a>
    </li>
    <li>
      <a href="#project-plan">Project plan</a>
    </li>
    <li>
      <a href="#the-team">The Team</a>
    </li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
Our team **Voyage SmartCar** from [Strive School](https://strive.school/), is working on building efficient smartcars to compete in the smart driving auto space. 
As part of the ongoing **AI Engineering specialization program**, this project is geared towards applying knowledge from computer vision module to solving a real life challenge.
This Smartcar driving challenge is what we are testing our skills on.


<!-- GETTING STARTED -->


### Goal of this project
* Our goal is to build a computer vison model on the opencv library that accurately identify roadlanes while driving as well as avoiding accident to the bearest minimum. 

### Installation and system setup
This project was built on the anaconda package, and the following installation setup was used for the succesful completion of the project.

* Opencv: An open comtuter vision library---- [conda install -c conda-forge opencv]
* Numpy: a scientific computing library from python---- [conda install numpy] 
* Matplotlib: A plot library for python---- [conda install -c conda-forge matplotlib]


## Project plan

1. Download and pre-process images associated with car driving lanes 
    * Take a video as input. Convert each frame into grayscale, apply a blur and use Canny for line detection.
2. Define a region of interest. Create a polygon that will sit on our lane ('field of vision') to negate background noise.  
    * The region of interest is the space around the vehicle where we track the car lanes. 
3. Isolate the lines found within our region of interest by applying the ROI as a mask to the frame that has been converted to contain lines.
4. Within the ROI mask find the shapes that are like road markings (Continous rectangle) or multiple rectangles (Dashed). 
    * Highlight these areas with a color.
5. prepare the video by specifying the frames per seconds, then get all frames  with the detected lanes together into a list, and finally combining the frames into a video.
6. Fill areas between lines to avoid accident to the bearest minimum.
7. Inspect the script by testing more different videos



<!-- CONTACT -->
## The Team:

| Contributors | Tasks | LINKEDIN|
| ------ | ------ | ------ |
| Sven Skyth Henriksen | model deployment| [https://www.linkedin.com/in/sven-skyth-henriksen-4857bb1a2/]|
| Mark Skinner | Model development | [https://www.linkedin.com/in/mark.skinner] |
| Daniel Biman | Model development | [https://www.linkedin.com/in/daniel-biman-05806a185/]|
| Olatunde Salami  | Project Manager | [https://www.linkedin.com/in/olatunde-salami/]|
