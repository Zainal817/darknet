#!/bin/bash
cd
sudo apt install pyqt5-dev-tools
python3 -m pip install lxml
git clone https://github.com/tzutalin/labelImg
cd labelImg
make qt5py3
python3 labelImg.py
