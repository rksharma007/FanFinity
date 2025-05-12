
# FanFinity🪭👽
Multi-matrix fan automation system based on human detection.

### Requirements
- opencv
- deploy.prototxt
- mobilenet_iter_73000.caffemodel
## Steps
1. Run autofan.py
2. In the window opened, mark the bounding box areas for different fan zones.
3. If a human is detected for more than 5 seconds in any box, that fan will turn on. It will turn off automatically if the human leaves that area for more than 3 seconds. (These threshold values can be configured.)
4. To quit (not in life) hit '*q*'.


![Screenshot 2025-05-12 154239](https://github.com/user-attachments/assets/163aa886-2694-4802-be07-2008d9a77cd3)
![Screenshot 2025-05-12 154007](https://github.com/user-attachments/assets/7c01dfb9-0f26-4426-aba5-f420cf575bab)
