# PatientX.AI

PatientX.AI is a tool designed to analyze and visualize patient experiences along specific treatment pathways. By scraping patient forums, running topic modeling algorithms, and constructing a "Journey map" based on these topics, PatientX.AI provides insights into common challenges, symptoms, and milestones that patients may encounter during their treatment. This tool aims to enhance both patient understanding of treatment processes and doctors' ability to provide timely support and resources.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Strucutre](#project-structure)
- [Installation](#installation)
- [Legacy R Pipeline](#legacy-r-pipeline)

## Overview

PatientX.AI offers:
- **Insight for Patients:** A visual map of common experiences to help patients prepare for different stages of their treatment.
- **Support for Medical Providers:** Analysis to help doctors identify critical stages where patients might benefit from additional resources, advice, or interventions.

## Features

- **Topic Modeling:** Uses BERTopic to identify common themes and group them into topics related to the treatment journey.
- **Topic Visualization:** Generates visualizations to help experts interpret topics identified through topic modeling
- **Journey Map Visualization:** Generates a user-friendly journey map, visually outlining typical patient experiences and symptom patterns.

## Project Structure
```
PatientX.AI/
├── script/install.sh                         # script to replace install.sh from ollama repo to install ollama rootless using --user flag
├── data/                                     # folder to hold data to be used by pipelines
├── python-notebook/                          # hold python notebook and documentation
├── existing_code/                            # folder to hold code prior to work with GT CSSE
│   └──legacy_Data_sempling_topic_modeling.r  # R script for running pipeline
└── src                                       # source code for PatientX.AI pipeline
└── tests                                     # tests
└── README.md     
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PatientX.AI.git
   ```

## Next Steps
For detailed information on how to run the PatientX.AI pipeline on your custom data, please check [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)

This file will go through steps to:
- Create and Activate a Virtual Environment
- Install Dependencies
- Setup for using LLMs for topic interpretation
- Run Pipelines


## Legacy R Pipeline
To run the legacy R pipeline, navigate to the `existing_code` folder and execute:
```bash
$> Rscript legacy_Data_sempling_topic_modeling.r
```

**Note:** Before running the R pipeline, ensure that the required R packages are installed. Refer to the comments in the `legacy_Data_sempling_topic_modeling.r` file for package installation instructions.
