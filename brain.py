from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path,'./images/godzilla.jpg'), result_count=5)
for eachPred, eachProb in zip(predictions, probabilities):
    print(f'{eachPred} : {eachProb}')