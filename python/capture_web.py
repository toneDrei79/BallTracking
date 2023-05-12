import cv2
import time


def capture(frame):
    cv2.imwrite('images/009.png', frame)


# カメラの読込み
# 内蔵カメラがある場合、下記引数の数字を変更する必要あり
cap = cv2.VideoCapture(0)

# 動画終了まで、1フレームずつ読み込んで表示する。
start_time = time.time()
while(cap.isOpened()):
    # 1フレーム毎　読込み
    ret, frame = cap.read()

    # GUIに表示
    cv2.imshow("Camera", frame)

    elapsed_time = int(time.time() - start_time)
    if elapsed_time > 10:
        capture(frame)
        break

    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        capture(frame)
        break
        

# 終了処理
cap.release()
cv2.destroyAllWindows()