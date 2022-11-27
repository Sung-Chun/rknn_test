from rknn.api import RKNN

import cv2
import numpy as np

model_file='/home/sungmin/WORK/train/EXP_320/weights/torchscript.pt'
input_size_list = [[1, 3, 640, 640]]


def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def RKNN_Test():
    # Create RKNN object
    rknn = RKNN(verbose=True)

    rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 58.395, 58.395])

    # Load model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model_file, input_size_list=input_size_list)
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    print('done')

    # Build model
    print('--> Building model') 
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model') 
    ret = rknn.export_rknn('./yolov5_lpr.rknn')
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    # Set inputs
    img = cv2.imread('./16422991740871249.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model') 
    outputs = rknn.inference(inputs=[img])
    np.save('./pytorch_yolov5_lpr.npy', outputs[0])
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()


if __name__ == '__main__':
    RKNN_Test()


