# MobileSAM — Full Installation Guide (Windows + Python 3.10)

1. Install Python 3.10 from python.org and make sure it is added to PATH.

2. Install Git.

3. Open PowerShell in your working directory.

4. Clone the MobileSAM repository:
git clone https://github.com/ChaoningZhang/MobileSAM.git

5. Create a Python 3.10 virtual environment (replace <User> with your username):
& "C:\Users<User>\AppData\Local\Programs\Python\Python310\python.exe" -m venv mobile_sam_env

6. Activate the environment:
.\mobile_sam_env\Scripts\activate


7. Install dependecies:
```
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install timm opencv-python pillow numpy matplotlib svgwrite ultralytics cadquery
```

8. Download the MobileSAM model weights (not included in the repo):
Invoke-WebRequest -Uri "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
" -OutFile mobile_sam.pt 

8. Create folders 
```bash
mkdir processed_png raw_png processed_svg;
```

9. Place your input your bottle .png's in a raw_png folder

10. Run:
 ```
 python process_to_png.py
 ```

11. You will get `filename_processed.png` files in your processed_png folder, with transparent background.

12. If you want to convert them to .svg run

 ```
 python convert_to_svg.py
 ```

13. You will get `filename_detailed.svg` files in your processed_svg folder.

14. Any time you want to use MobileSAM, activate the environment:
 ```
 .\mobile_sam_env\Scripts\activate