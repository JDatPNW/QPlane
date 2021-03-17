
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/JDatPNW/QPlane">
    <img src="misc/QPlaneLogo.svg" alt="Logo" width="80%">
  </a>

  <h3 align="center">QPlane</h3>

  <p align="center">
    XPlane flight simulation controlled by a QLearning / Deep QLearning Algorithm
    <br />
    <br />
    <a href="https://github.com/JDatPNW/QPlane/issues">Report Bug</a>
    ·
    <a href="https://github.com/JDatPNW/QPlane/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository is being written as part of my masters thesis. I am trying to develop a fixed wing attitude control system using Reinforcement Learning algorithms. As of right now this code works with XPlane 11 and QLearning as well as Deep QLearning.

### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [tensorflow](https://www.tensorflow.org/)
* [XPlaneConnect](https://github.com/nasa/XPlaneConnect)
* [XPlane 11](https://www.x-plane.com/)
* [JSBSim](https://github.com/JSBSim-Team/jsbsim)
* [Flightgear](https://www.flightgear.org/)



<!-- GETTING STARTED -->
## Getting Started

Simple clone this repository to your local filesystem:
```sh
git clone https://github.com/JDatPNW/QPlane
```

### Prerequisites

Tested and running with:

|Software | Version|
|-----|-----|
|XPlane11 Version: | 11.50r3 (build 115033 64-bit, OpenGL)|
|JSBSim Version: | 1.1.5 (GitHub build 277)|
|Flightgear Version: | 2020.3.6|
|XPlaneConnect Version: | 1.3-rc.2|
|Python Version: | 3.8.2|
|numpy Version: | 1.19.4|
|tensorflow Version: | 2.3.0|
|Windows Version: | 1909|


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install the above listed software (other versions might work)
  * For JSBSim clone the JSBsim repo into `src/environments/jsbsim`
  * For visualizing JSBSim download the c172r plane model in the Flightgear Menu

<!-- USAGE EXAMPLES -->
## Usage

Once downloaded and installed, simply execute the `QPlane.py` file to run and test the code.
* For the XPlane Environment, XPlane (the game) needs to run.
* For JSBSim with rendering, Flightgear needs to run with the following flags `--fdm=null --native-fdm=socket,in,60,localhost,5550,udp --aircraft=c172r --airport=RKJJ`

<!-- ROADMAP -->
## Roadmap

Planned future features are:
* PPO implementation
* JSBSim implementation (currently working on)


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `misc/LICENSE` for more information.



<!-- CONTACT -->
## Contact

Github Pages: [JDatPNW](https://github.com/JDatPNW/JDatPNW.github.io)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Readme Template](https://github.com/othneildrew/Best-README-Template)
* [Python Programming DeepRL](https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/)
* [Deeplizard DeepRL](https://deeplizard.com/learn/playlist/PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv)
* [adderbyte](https://github.com/adderbyte/GYM_XPLANE_ML)
* [XPlane Forum](https://forums.x-plane.org/index.php?/forums/topic/236878-xplane11-xplaneconnect-question-about-resettingspawning-the-plane/&tab=comments#comment-2118006)
* [JSBSim](https://github.com/JSBSim-Team/jsbsim)
