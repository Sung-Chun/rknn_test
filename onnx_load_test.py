##
# It is to test onnxruntime.
# It loads ONNX_MODEL and do the onnxruntime.InferenceSession for randomly generated input
##

import cv2
def img_test():
    '''
    img_test() loads image from file, resizes it and then write to file.
    '''
    img_filename = './rknpu2_test/test_images/63_4654.png'
#    img_filename = './16422991588116416.jpg'

    # Set inputs
    img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)

#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (160, 80))
    cv2.imwrite('./160x80x1.jpg', img)
    pass


import torch
from torch import nn
import onnx
import onnxruntime

if True:
    MODEL_INPUT_SIZE = (1, 1, 768, 1024)   #(batch_size, num channels, height, width)
    MEAN_VALUES = [0]
    STD_VALUES = [256]
    ONNX_MODEL = './ONNX_V12_TrainedModels/Model-1/train-1_23_028-model-iter_1507.onnx'
    RKNN_MODEL = 'train-1_23_028-model-iter_1507.rknn'
else:
    MODEL_INPUT_SIZE = (1, 3, 768, 1024)
    MEAN_VALUES = [0, 0, 0]
    STD_VALUES = [256, 256, 256]
    ONNX_MODEL = '/home/sungmin/WORK/train/EXP_320/weights/best.onnx'
    RKNN_MODEL = 'best.rknn'

def onnx_test():
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_model = onnx.load(ONNX_MODEL)
    onnx.checker.check_model(onnx_model)

    batch_size = MODEL_INPUT_SIZE[0]
    x = torch.randn(batch_size, MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[2], MODEL_INPUT_SIZE[3], requires_grad=True)

    ort_session = onnxruntime.InferenceSession(ONNX_MODEL)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(type(ort_outs))
    pass


if __name__ == '__main__':
    img_test()
#    onnx_test()
