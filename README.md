# Modern AI Pro

Welcome to the Modern AI Pro repository! This repository contains a collection of Python scripts designed to help students explore and learn various aspects of machine learning, artificial intelligence, and model management using MLFlow. Each script in the repository is a standalone sample that demonstrates specific concepts or techniques.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Scripts Overview](#scripts-overview)
- [Running the Scripts](#running-the-scripts)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

To set up your environment to run these scripts, follow these steps:

1. **Clone the repository**: Begin by cloning the repository to your local machine using Git. Open your terminal or command prompt and run the following command:
   ```bash
   git clone https://github.com/balajivis/modernaipro.git
   ```

2. **Install Visual Studio Code** (VS Code): 
   If you haven't already installed Visual Studio Code, you can download and install it from [here](https://code.visualstudio.com/download). VS Code is our recommended code editor for working with this repository.

3. **Install Dev Container extension** in VS Code: 
   
   To streamline development within a containerized environment, we recommend installing the Dev Container extension in Visual Studio Code. Follow these steps:
   - Open Visual Studio Code.
   - Go to the Extensions view by clicking on the square icon in the sidebar or pressing `Cmd+Shift+X`(MacOS) or `Ctrl+Shift+X`(Others).
   - Search for "Dev Container" in the Extensions Marketplace.
   - Install the "Dev Container" extension by clicking on the Install button.

4. **Open Folder in VS Code**: 
   - After installing VS Code and the Dev Container extension, open VS Code and use it to navigate to the repository directory.
   - You can do this by either running the command `code .` in the terminal or opening VS Code first and then using the UI to open the folder `modernaipro`.

5. **Open the Dev Container**:
   - Click on the green icon in the bottom-left corner of Visual Studio Code (or press Cmd+Shift+P (MacOS) or Ctrl+Shift+P (Others))
   - Search for "Dev Containers: Reopen in Container" on the top search bar.
   - This action will open the project inside the Dev Container.

### Installing Models using Ollama

In the VS Code container. You can use the Ollama extension to install models. We will be using the following models in this repository:
- phi3
- qwen
- llava

To download the models. Follow these steps:
- Open terminal in the VS Code (^ + `)
- run `ollama pull phi3`
- This will take some time to download the whole model. 
- Similarly install other models.

