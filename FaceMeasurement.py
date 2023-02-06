import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from imutils.video import FPS

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
detector = FaceMeshDetector(maxFaces=1)

fps = FPS().start()
while True:
    success, img = cap.read()
    dsize = (1920, 1080)
    res_img = cv2.resize(img, dsize)
    res_img, faces = detector.findFaceMesh(res_img)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        cv2.circle(res_img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(res_img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        img = cv2.line(res_img, pointLeft, pointRight, color=(0, 255, 0), thickness=5)
        # dis = math.sqrt(pow(pointLeft[0] - pointRight[0], 2) - pow(pointLeft[1] - pointRight[1], 2))
        dis, _ = detector.findDistance(pointLeft, pointRight)
        dis = np.around(dis, 3)

        W = 6.3
        # Finding the Focal length
        # d = 21
        # f = (dis*d)/W

        # Finding the distanse
        f = 1800
        d = (W * f) / dis
        # print(d)

        cv2.putText(res_img, f'Depth: {int(d)} sm', pointLeft, 0, 2, (255, 255, 255),
                    3)  # text,coordinate,font,size of text,color,thickness of font
    fps.update()
    cv2.imshow("output", res_img)
    if cv2.waitKey(5) == 27:
        print("Окончание программы по обнаружению движущихся обьектов")
        break
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()
