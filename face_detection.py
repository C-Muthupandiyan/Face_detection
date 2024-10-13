import cv2 #import opencv
#haarcascade dataset for face detection
trainedDataSet= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0) #it is for accesing your webcam


while True:
    success, frame = video.read() #read a webcam live video
    if success==True:  # Checking if the frame was successfully captured
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert into black white video
        faces = trainedDataSet.detectMultiScale(gray_image) #insert dataset
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)#draw rectangle on face
        cv2.imshow('face', frame)#show video
        key = cv2.waitKey(1)  # it is for hold webcam live video
    if key==27:
        break


video.release()# stop webcam
cv2.destroyAllWindows()#it will destroy all run window.
