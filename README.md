# 👁️‍🗨️ Computer Vision Hub

> **The Frontier of Computational Perception.**  
> A unified Operations Center showcasing projects focused on image processing, object detection, and interactive recognition using OpenCV, MediaPipe, and Deep Learning.

<div align="left">
  <a href="https://www.linkedin.com/in/vitoriapguimaraes/"><img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="mailto:vipistori@gmail.com"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Gmail"/></a>
  <a href="https://github.com/vitoriapguimaraes/vitoriapguimaraes/blob/main/RESUME.md"><img src="https://img.shields.io/badge/Resume%20(Markdown)-gray?style=for-the-badge" alt="Resume ATS md"/></a>
  <a href="https://github.com/vitoriapguimaraes/vitoriapguimaraes/blob/main/doc/Vitoria_Pistori_RESUME.pdf"><img src="https://img.shields.io/badge/Resume%20(PDF)-gray?style=for-the-badge" alt="Resume ATS PDF"/></a>
</div>

![System Demonstration](https://github.com/vitoriapguimaraes/computerVision/blob/main/streamlit_app/assets/demos/streamlit_app_painel_screen.png)

## Table of Contents

- [Tech Stack](#tech-stack)
- [Main Features](#main-features)
- [How to Run](#how-to-run)
- [Directory Structure](#directory-structure)
- [Status & Roadmap](#status--roadmap)
- [Areas of Expertise](#areas-of-expertise)
- [Academic and Continuous Learning](#academic-and-continuous-learning)

## Tech Stack

![Python](https://img.shields.io/badge/-Python-000000?style=for-the-badge&logo=python&logoColor=white)
![Computer Vision](https://img.shields.io/badge/-Computer%20Vision-000000?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-000000?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-000000?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-000000?style=for-the-badge&logo=streamlit&logoColor=white)

## Main Features

- **Image Classification**: Image classification based on a CNN trained on the CIFAR-10 dataset.
- **Traffic Analysis**: Automatic vehicle counting using background subtraction algorithms.
- **Human-Machine Interaction**: Touchless interfaces utilizing real-time hand landmark detection.
- **Road Safety**: Driver drowsiness detection by monitoring the Eye Aspect Ratio (EAR).
- **Centralized Interface**: All algorithms run from a single interactive Streamlit dashboard in a "CCTV" style.

## How to Run

> ⚠️ **IMPORTANT: Use Docker!**  
> This project uses WebRTC and PyAV for real-time browser video streaming. Installing these libraries directly on Windows via `pip` usually fails because it requires compiling C++ FFmpeg libraries. **Running via Docker is strongly recommended.**

1. Clone the repository:

   ```bash
   git clone https://github.com/vitoriapguimaraes/dataScience.git
   cd dataScience/computerVision
   ```

2. **Start the Docker Container (Recommended):**

   ```bash
   # Ensure Docker Desktop is running on your machine
   docker compose up --build
   ```

3. **Local Setup (Linux / Mac / Advanced Windows Users):**
   *If you cannot use Docker, ensure you have Python 3.10 and FFmpeg development headers installed before running pip.*

   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
   
   pip install -r requirements.txt
   streamlit run streamlit_app/Painel.py
   ```

**Usage:** The hub will automatically be available in your browser at `http://localhost:8501` (if running locally) or `http://127.0.0.1:8501` (if using Docker on Windows). Navigate through the tabs in the sidebar to access the different computer vision tools.

## Directory Structure

The individual folders for the original CV projects are available in `projects/`, but their interactive logic is now integrated into the central application.

```dash
computerVision/
├── projects/                        # Original logic and isolated scripts (7 projects)
├── streamlit_app/
│   ├── assets/                      # Demonstration images and GIFs
│   ├── models/                      # Trained model weights (e.g., H5)
│   ├── pages/                       # Central Hub Pages
│   │   ├── 1_Image_Classification.py
│   │   ├── 2_Traffic_Analysis.py
│   │   ├── 3_Human_Machine_Interaction.py
│   │   └── 4_Road_Safety.py
│   ├── utils/                       # Shared components and utilities
│   │   ├── ui.py
│   │   ├── config.py
│   │   └── hand_tracking.py
│   └── Painel.py                    # Operations Dashboard (Home)
├── requirements.txt                 # Hub Dependencies (Requires Python 3.10)
└── README.md
```

## Status Legend

- ✅ **Completed**: Functional current version delivered.
- 🛠️ **In Maintenance**: Adjustments and corrections in progress.
- 🚧 **In Development**: New functionalities being implemented.

## Status & Roadmap

✅ **Completed**: The core Streamlit Hub integrating the main computer vision models is fully operational.
🚧 **In Development**:

- **Project 7 (Face Recognition):** Currently under construction and not yet integrated into the Streamlit Hub.

## Areas of Expertise

This repository is part of my larger developer portfolio:

- **Data Science & AI:** Data analysis, machine learning models, visualization, and intelligent agents. Check out my work in the [Data Science Portfolio](https://github.com/vitoriapguimaraes/portifolio-dataScience).
- **Full Stack Development:** Building web interfaces, automations, backend integrations, and robust applications. Explore my specialized projects in the [Full Stack Portfolio](https://github.com/vitoriapguimaraes/portfolio-developerFullStack).

## Academic and Continuous Learning

- **B.S. in Systems Analysis and Development (A.D.S.)** | Descomplica Digital University
- **Advanced Data Specialization** | Laboratoria
- **M.Sc. in Sciences (Data Analysis & Statistics Focus)** | University of São Paulo (USP)

[![View Academic Repo](https://img.shields.io/badge/Explore%20My%20Academic%20Repositories-gray?style=for-the-badge)](https://github.com/vitoriapguimaraes/ADS)

---

Learn more about my qualifications and certifications in the [Documents Folder](https://github.com/vitoriapguimaraes/vitoriapguimaraes/tree/main/doc).

> Questions, suggestions, or interested in collaborating? Feel free to contact me!
