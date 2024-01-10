import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay
import pickle
import os

def report(y_true,outputs,path):
  RocCurveDisplay.from_predictions(
    y_true.ravel(),
    outputs.ravel(),
    name="micro-average OvR",
    color="darkorange",
    plot_chance_level=True,
  )
  class_names = ["bend",
    "crouch",
    "sit",
    "stand",
    "walk",
    "drive",
    "lift",
    "pull",
    "push",
    "put down",
    "overhead lift",
    "reaching"]
  
  labels = [1,2,3,4,5,6,7,8,9,10,11,12]
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
  plt.legend()
  plt.savefig(os.path.join(path,'ROC.png'))
  plt.close()
  plt.figure().set_figheight(20)
  cm = confusion_matrix(np.argmax(y_true,axis = -1),np.argmax(outputs,axis = -1),labels = labels)
  cm = ConfusionMatrixDisplay(cm,display_labels=class_names).plot()
  plt.xticks(rotation='vertical')
  plt.gcf().set_size_inches(10, 10)
  plt.savefig(os.path.join(path,'confusion_matrix.png'))

if __name__ == '__main__':
  OUTPUT_PATH = "/home/ubuntu/Rumman/mmaction2/work_dirs/slow_fast_test_weller/results"
  file = open('/home/ubuntu/Rumman/mmaction2/work_dirs/slow_fast_test_weller/test.pkl', 'rb')
  data = pickle.load(file)
  file.close()
  outputs = []
  y_true= []
  for i,sample in enumerate(data):
    for output,label in zip(sample["pred_instances"]["scores"],sample["gt_instances"]["labels"]):
      outputs.append(output.numpy())
      y_true.append(label.numpy())
  report(np.array(y_true),np.array(outputs),OUTPUT_PATH)