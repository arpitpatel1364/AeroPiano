import cv2

def list_cameras():
    print("Scanning for cameras...")
    index = 0
    arr = []
    while index < 5:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {index}: Camera found and working.")
                arr.append(index)
            else:
                print(f"Index {index}: Found, but could not read frame.")
            cap.release()
        else:
            print(f"Index {index}: Not found.")
        index += 1
    return arr

if __name__ == "__main__":
    list_cameras()
