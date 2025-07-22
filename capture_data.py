import cv2
import os
import numpy as np
import time
from keras.models import load_model

image_path = r"images"
result_dict = {0:'Early_Blight', 1:'Healthy', 2:'Late_Blight'}

model_path = "model.keras"

def preprocess(path):
    path = os.listdir(path)[0]
    path_new = os.path.join(image_path,path)
    print("image saved")
    img = cv2.imread(path_new)
    img = cv2.resize(img,(50,50))
    img = np.array(img)/255
    img = np.expand_dims(img,axis=0)

    return img


def get_img():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret , frame = cap.read()
        cv2.imwrite("images\data.jpg",frame)
        cv2.putText(frame,"please do this on *white paper",(50,30),cv2.FONT_HERSHEY_TRIPLEX,1,(32,43,230),4,cv2.LINE_AA)
        cv2.putText(frame, 'for shoot press "q" or Exit press "p"', (50,70), cv2.FONT_HERSHEY_COMPLEX, 1, (120,80, 20), 4, cv2.LINE_AA)
        cv2.imshow("Live",frame)
       
        print("images on capturing")

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            IMG =preprocess(image_path) 
            break
        elif cv2.waitKey(10) &0xFF==ord("p"):
            IMG = None
            break
    
    cap.release()
    cv2.destroyAllWindows()   
    return IMG
 


def predict():
    try:
        img = get_img()
        model = load_model(model_path)
        result = model.predict(img)
        result = np.argmax(result)
        x = result_dict[result]

        img = cv2.imread(r"images\data.jpg")
        if x=="Healthy":
            cv2.putText(img,f"Your plants are {x}",(50,40),cv2.FONT_HERSHEY_SIMPLEX,1.3,(120,80, 20),4,cv2.LINE_AA)

        else:
            cv2.putText(img,f"Your plants has {x}",(50,40),cv2.FONT_HERSHEY_SIMPLEX,1.3,(120,80, 20),4,cv2.LINE_AA)

        img = cv2.imshow("Live",img)
        cv2.waitKey(0)

    except :
        x = "None"


if __name__ == "__main__":
    predict()

