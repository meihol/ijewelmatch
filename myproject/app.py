import time
from datetime import datetime

FILE_PATH = "/data/hello.txt"

while True:
    with open(FILE_PATH, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Hello World\n")
    print("Wrote line to", FILE_PATH)
    time.sleep(5)  # write every 5 seconds
